"""Prompt centering test: subtract each prompt's average features so topic cancels and style remains.

Out-of-fold over folds 0-3 (never fold 4), frozen run3 features (PCA 256, fitted on pool rows only), logistic head.
  raw       : head on raw features
  centered  : train rows centred by their true prompt; validation rows centred by prompt groups FOUND by clustering
              (as the hidden test would be), then the head on [raw, centred]
Two validation conditions:
  full      : each prompt keeps all its passages (18, one per model; how train.csv is built)
  partial   : each prompt keeps a random 2-10 of them (the hidden test is NOT built like train.csv: the group
              assignment probe scored 0.367 vs 0.402), repeated over 3 seeds
Also: logit-level centering of the shipped blend (its OOF log-probs minus w x the cluster mean).
Run: .venv/Scripts/python.exe features/centering.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from assign import find_groups, log_softmax  # noqa: E402

P = ROOT / "data/processed"
train = pd.read_csv(ROOT / "data/train.csv")
train["id"] = train.id.astype(str)
fold_of = {}
for k, (_, va) in enumerate(GroupKFold(5).split(train, train.model, train.prompt_id)):
    fold_of.update({i: k for i in train.id.iloc[va]})
bl = np.load(P / "blend_oof.npz", allow_pickle=True)
ids = [str(i) for i in bl["ids"]]
lp, y = bl["lp"], bl["y"]
meta = pd.read_parquet(P / "feats_run3_host.parquet")
X = np.load(P / "feats_run3_host.npy").astype(np.float32)[
    pd.Series(np.arange(len(meta)), index=meta.id.astype(str)).loc[ids].to_numpy()]
tr_tab = train.set_index("id").loc[ids]
prompt, text = tr_tab.prompt_id.to_numpy(), tr_tab.text.fillna("").to_numpy()
fold = np.array([fold_of[i] for i in ids])
pX = np.load(P / "feats_run3_pool.npy").astype(np.float32)
sc = StandardScaler().fit(pX)
pca = PCA(256, random_state=0).fit(sc.transform(pX))
Z = pca.transform(sc.transform(X))


def center(F, groups):
    out = F.copy()
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= 2:
            out[m] -= F[m].mean(0)
    return out


def run(keep):
    """keep: boolean mask of validation rows that exist in this simulated test."""
    raw_p = np.full_like(lp, np.nan)
    cen_p = np.full_like(lp, np.nan)
    blend_c = {w: np.full_like(lp, np.nan) for w in (0.25, 0.5, 1.0)}
    for k in range(4):
        tr, va = (fold != k), (fold == k) & keep
        idx = np.flatnonzero(va)
        groups = find_groups(text[idx]) if len(idx) > 36 else np.zeros(len(idx), int)
        Ztr, Zva = Z[tr], Z[idx]
        Ctr, Cva = center(Ztr, prompt[tr]), center(Zva, groups)
        s1 = StandardScaler().fit(Ztr)
        raw_p[idx] = LogisticRegression(C=0.05, max_iter=3000).fit(s1.transform(Ztr), y[tr]).predict_log_proba(s1.transform(Zva))
        F_tr, F_va = np.hstack([Ztr, Ctr]), np.hstack([Zva, Cva])
        s2 = StandardScaler().fit(F_tr)
        cen_p[idx] = LogisticRegression(C=0.05, max_iter=3000).fit(s2.transform(F_tr), y[tr]).predict_log_proba(s2.transform(F_va))
        for w in blend_c:
            cluster_mean = lp[idx] - center(lp[idx], groups)  # zero for clusters of one
            blend_c[w][idx] = lp[idx] - w * cluster_mean
    m = keep
    res = {"frozen head raw": bacc(y[m], raw_p[m].argmax(1)), "frozen head raw+centred": bacc(y[m], cen_p[m].argmax(1)),
           "blend": bacc(y[m], lp[m].argmax(1))}
    res.update({f"blend - {w} x cluster mean": bacc(y[m], blend_c[w][m].argmax(1)) for w in blend_c})
    return res


print("FULL prompt groups (18 per prompt):")
for k, v in run(np.ones(len(y), bool)).items():
    print(f"  {k:28s} {v:.4f}")
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
for k, v in acc.items():
    print(f"  {k:28s} {np.mean(v):.4f}  (seeds: {', '.join(f'{x:.3f}' for x in v)})")
