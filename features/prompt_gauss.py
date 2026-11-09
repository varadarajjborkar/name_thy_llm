"""Component 4: shared-prompt Gaussian group head.

Two-way model: x_i = mu[class_i] + u[prompt] + e_i,  u ~ N(0, T) shared by a prompt's passages, e ~ N(0, S).
Fitted on training rows (known prompts): mu = class means, S = within-prompt noise, T = prompt-shift covariance.
S and T are diagonalised together (S = I, T = diag(lam)), so the posterior of the shared shift u is per dimension:
  u_hat = sum_i (x_i - E[mu]) * lam / (1 + n * lam)       (n passages in the group; shrinks towards 0 for small n)
Test rows are grouped by prompt clusters FOUND from text (as on the hidden test); a few EM rounds alternate
soft class posteriors and u_hat. Output: per-row class log-likelihoods after removing u_hat.

Feature spaces: (a) the blend's own OOF log-probs (18 dims; shippable as a post-step), (b) frozen run3 features
(PCA 128, fitted on pool rows). Out of fold on folds 0-3; the fit for fold k uses rows outside fold k.
Compared against the shipped post-processing: blend - 0.5 x cluster mean, then partial Sinkhorn 0.45.
Two conditions: full prompt groups (18 per prompt) and partial (random 2-10 per prompt, 3 seeds).
Run: .venv/Scripts/python.exe features/prompt_gauss.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from assign import find_groups, log_softmax, sinkhorn  # noqa: E402

P = ROOT / "data/processed"
train = pd.read_csv(ROOT / "data/train.csv")
train["id"] = train.id.astype(str)
fold_of = {}
for k, (_, va) in enumerate(GroupKFold(5).split(train, train.model, train.prompt_id)):
    fold_of.update({i: k for i in train.id.iloc[va]})
d = np.load(P / "blend_parts.npz", allow_pickle=True)
ids, y = [str(i) for i in d["ids"]], d["y"]
lp = log_softmax(0.5 * d["s4"] + 0.3 * d["s2mix"] + 0.1 * d["svm"] + 0.1 * d["lgb"])  # shipped v6 blend
tab = train.set_index("id").loc[ids]
prompt, text = tab.prompt_id.to_numpy(), tab.text.fillna("").to_numpy()
fold = np.array([fold_of[i] for i in ids])
K = lp.shape[1]

meta = pd.read_parquet(P / "feats_run3_host.parquet")
X = np.load(P / "feats_run3_host.npy").astype(np.float32)[
    pd.Series(np.arange(len(meta)), index=meta.id.astype(str)).loc[ids].to_numpy()]
pX = np.load(P / "feats_run3_pool.npy").astype(np.float32)
sc = StandardScaler().fit(pX)
Zfeat = PCA(128, random_state=0).fit(sc.transform(pX)).transform(sc.transform(X))
del X, pX


class PromptGauss:
    def fit(self, F, yy, pp, ridge=1e-3):
        self.mu = np.stack([F[yy == c].mean(0) for c in range(K)])
        R = F - self.mu[yy]
        pm = pd.DataFrame(R).groupby(pp).transform("mean").to_numpy()
        n_p = pd.Series(pp).map(pd.Series(pp).value_counts()).to_numpy()
        E = R - pm  # within-prompt noise
        S = np.cov(E.T) * (n_p.mean() / max(n_p.mean() - 1, 1)) + ridge * np.eye(F.shape[1])
        G = pd.DataFrame(R).groupby(pp).mean().to_numpy()  # prompt means of residuals = u + noise / n
        T = np.cov(G.T) - S / n_p.mean()
        lam, W = eigh(T, S)  # W^T S W = I, W^T T W = diag(lam)
        self.lam, self.W = np.clip(lam, 0, None), W
        self.m = self.mu @ W
        return self

    def loglik(self, F, groups=None, iters=5, prior=None):
        Z = F @ self.W
        lp0 = np.zeros((len(F), K)) if prior is None else prior
        u = np.zeros_like(Z)
        ll = -0.5 * ((Z[:, None, :] - self.m[None]) ** 2).sum(-1)
        if groups is None:
            return ll
        for _ in range(iters):
            q = np.exp(log_softmax(ll + lp0))
            r = Z - q @ self.m
            for g in np.unique(groups):
                ix = groups == g
                n = ix.sum()
                u[ix] = r[ix].sum(0) * self.lam / (1 + n * self.lam) if n >= 2 else 0
            ll = -0.5 * (((Z - u)[:, None, :] - self.m[None]) ** 2).sum(-1)  # with the newest shift estimate
        return ll


def center(F, groups, w):
    out = F.copy()
    for g in np.unique(groups):
        ix = groups == g
        if ix.sum() >= 2:
            out[ix] -= w * F[ix].mean(0)
    return out


def psink(z, a=0.45):
    z = log_softmax(z)
    return (1 - a) * z + a * sinkhorn(z)


def run(keep):
    out = {"blend": np.full_like(lp, np.nan), "blend+centre": np.full_like(lp, np.nan)}
    for name in ("G-lp", "G-feat"):
        for s in ("nogroup", "group"):
            out[f"{name} {s}"] = np.full_like(lp, np.nan)
    for k in range(4):
        tr, idx = np.flatnonzero(fold != k), np.flatnonzero((fold == k) & keep)
        groups = find_groups(text[idx]) if len(idx) > 36 else np.zeros(len(idx), int)
        out["blend"][idx] = lp[idx]
        out["blend+centre"][idx] = center(lp[idx], groups, 0.5)
        for name, F in (("G-lp", lp), ("G-feat", Zfeat)):
            g = PromptGauss().fit(F[tr], y[tr], prompt[tr])
            out[f"{name} nogroup"][idx] = g.loglik(F[idx])
            out[f"{name} group"][idx] = g.loglik(F[idx], groups)
    m = keep
    res = {}
    for key, z in out.items():
        res[key] = (bacc(y[m], z[m].argmax(1)), bacc(y[m], psink(z[m]).argmax(1)))
    base = out["blend+centre"][m]
    for key in ("G-lp group", "G-feat group"):
        z = log_softmax(out[key][m] / 1.0)
        for w in (0.2, 0.4):
            res[f"centred blend + {w} x {key}"] = (bacc(y[m], (base + w * z).argmax(1)),
                                                   bacc(y[m], psink(base + w * z).argmax(1)))
    return res


def show(rows):
    for k, v in rows.items():
        a = np.mean([x[0] for x in v]) if isinstance(v, list) else v[0]
        b = np.mean([x[1] for x in v]) if isinstance(v, list) else v[1]
        print(f"  {k:38s} argmax {a:.4f}   partial-sinkhorn {b:.4f}", flush=True)


print("FULL prompt groups (18 per prompt):")
show(run(np.ones(len(y), bool)))
acc = {}
for seed in range(3):
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(y), bool)
    for p in np.unique(prompt):
        rows = np.flatnonzero(prompt == p)
        keep[rng.choice(rows, size=rng.integers(2, 11), replace=False)] = True
    for k, v in run(keep).items():
        acc.setdefault(k, []).append(v)
print("PARTIAL prompt groups (random 2-10 per prompt, 3 seeds, mean):")
show(acc)
