"""A blend built around t4 (Qwen3-4B fine-tuned on train.csv only), with every other member also trained on train.csv
only, and its weights tuned for t4 rather than inherited from the s4-era blend. Folds 0-3 out of fold, never fold 4.

Question 1 (user, 2026-11-12): was t4 weak inside v7 only because the other members and weights were tuned to s4?
  -> v7's members (s2mix, SVM, ext LightGBM, ATLAS host+pool) with t4 in place of s4, weights RE-TUNED.
Question 2: the honest train-only system. Members (all fit on the training folds of train.csv only):
  t4        Qwen3-4B LoRA + multi-scale head (checkpoints/t4f0-3)
  svm4      4-view n-gram SVM, host rows only (research/svm4.py, data/processed/svm4_oof_host.npy)
  lgb       LightGBM on style features, train.csv CV (features/lgbm_run.py cv, lgbm_cv_oof.npy)
  atlas     ATLAS base-LM likelihood features + LightGBM, host rows only (no pool), class-balanced, 5 seeds
  letters   char_wb 2-5 TF-IDF + logistic regression
  trace_lr  TRACE fingerprints (arXiv 2603.28054): joint frequencies of consecutive (rank bucket, rank bucket),
            (entropy bin, entropy bin) and (surprise bin, surprise bin) pairs from Qwen3-4B base, + logistic regression
  trace_nn  the paper's own classifier: one reference fingerprint per LLM (mean of its training texts), score =
            -Jensen-Shannon distance to each reference
Weights: coordinate ascent on a 0.05 grid, NESTED (chosen on 3 folds, scored on the 4th) so the number is honest;
the in-sample optimum is printed too. Post-processing as the notebook: prompt centering 0.5 + partial sinkhorn 0.45.
SPARSE view (test-like, research/tree2_eval.py): per prompt 2-6 texts with an uneven Dirichlet model mix, the four
folds' subsets pooled (~480 rows, so the >= 180-row post rule is met), groups found on the subset only, 3 draws.
Run: .venv/Scripts/python.exe research/blend_trainonly.py
"""
import sys
import time
from pathlib import Path

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from assign import find_groups, log_softmax, sinkhorn  # noqa: E402

try:
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass
T0 = time.time()
LOG = ROOT / "logs/blend_trainonly.log"
LOG.write_text("", encoding="utf-8")


def say(*x):
    msg = " ".join(str(v) for v in x)
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


P = ROOT / "data/processed"
train = pd.read_csv(ROOT / "data/train.csv")
train["id"] = train.id.astype(str)
classes = sorted(train.model.unique())
cid = {c: i for i, c in enumerate(classes)}
fold_all = np.empty(len(train), dtype=int)
for k, (_, va) in enumerate(GroupKFold(5).split(train, train.model, train.prompt_id)):
    fold_all[va] = k
fold_of = dict(zip(train.id, fold_all))
d = np.load(P / "blend_parts.npz", allow_pickle=True)
ids, y = [str(i) for i in d["ids"]], d["y"]
fold = np.array([fold_of[i] for i in ids])
pos = pd.Series(np.arange(len(train)), index=train.id).loc[ids].to_numpy()  # blend row -> train.csv row
assert set(fold) == {0, 1, 2, 3} and (train.model.map(cid).to_numpy()[pos] == y).all()
text_all = train.text.fillna("").astype(str).to_numpy()
y_all = train.model.map(cid).to_numpy()
text = text_all[pos]


def family(prefix):
    parts = []
    for k in range(4):
        r = ROOT / f"checkpoints/{prefix}f{k}"
        assert __import__("json").load(open(r / "classes.json")) == classes
        vi = pd.read_csv(r / "val_ids.csv").id.astype(str)
        parts.append(pd.DataFrame(log_softmax(np.load(r / "val_logits.npy")), index=vi))
    f = pd.concat(parts)
    assert f.index.is_unique and set(f.index) == set(ids)
    return f.loc[ids].to_numpy()


M = {"t4": family("t4"), "s4": d["s4"], "s2mix": d["s2mix"], "svm_v7": d["svm"], "lgb_ext": d["lgb"]}
M["svm4"] = log_softmax(3 * np.load(P / "svm4_oof_host.npy"))
M["lgb"] = np.log(np.clip(np.load(P / "lgbm_cv_oof.npy")[pos], 1e-9, 1))

# ATLAS: host-only (train-only) and host+pool (as v7 ships), class-balanced, 5 seeds
H = pd.read_parquet(P / "atlas_host_fp16.parquet").assign(id=lambda x: x.id.astype(str))
Pl = pd.read_parquet(P / "atlas_pool_fp16.parquet")
cols = [c for c in H.columns if c not in ("id", "model")]
Hx = H.set_index("id").loc[ids, cols].to_numpy()
hfold = H.id.map(fold_of).to_numpy()
PRM = dict(objective="multiclass", num_class=len(classes), learning_rate=0.05, num_leaves=15, min_child_samples=20,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=6)
for name, with_pool in (("atlas", False), ("atlas_pool", True)):
    out = np.zeros((len(ids), len(classes)))
    for k in range(4):
        tr = hfold != k
        X, Y = H.loc[tr, cols].to_numpy(), H.model[tr].map(cid).to_numpy()
        if with_pool:
            X, Y = np.vstack([X, Pl[cols].to_numpy()]), np.concatenate([Y, Pl.model.map(cid).to_numpy()])
        w = (len(Y) / (len(classes) * np.bincount(Y, minlength=len(classes))))[Y]
        pr = np.mean([lgbm.train(dict(PRM, seed=s), lgbm.Dataset(X, Y, weight=w), num_boost_round=300)
                      .predict(Hx[fold == k]) for s in range(5)], axis=0)
        out[fold == k] = np.log(np.clip(pr, 1e-9, 1))
    M[name] = out
    say(f"{name} done {time.time() - T0:.0f}s")

# letter chunks
out = np.zeros((len(ids), len(classes)))
for k in range(4):
    tr = fold_all != k
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=60000)
    lr = LogisticRegression(C=2.0, max_iter=3000).fit(vec.fit_transform(text_all[tr]), y_all[tr])
    out[fold == k] = np.log(np.clip(lr.predict_proba(vec.transform(text[fold == k])), 1e-12, 1))
M["letters"] = out
say(f"letters done {time.time() - T0:.0f}s")

# TRACE fingerprints from the saved per-token sequences (features/signal_run.py, Qwen3-4B base fp16)
Z = np.load(P / "signal_train_seqs.npz", allow_pickle=True)  # ids are strings saved as an object array
zpos = pd.Series(np.arange(len(Z["ids"])), index=[str(i) for i in Z["ids"]]).loc[train.id].to_numpy()
seqs = [(Z[f"s_{j}"], Z[f"e_{j}"], Z[f"r_{j}"]) for j in zpos]  # in train.csv row order
RB = np.array([1, 3, 10, 100, 1000])  # rank buckets 0 | 1-2 | 3-9 | 10-99 | 100-999 | 1000+
NB = 5
e_cut = np.quantile(np.concatenate([q[1] for q in seqs]), np.linspace(0, 1, NB + 1)[1:-1])  # label-free bins
s_cut = np.quantile(np.concatenate([q[0] for q in seqs]), np.linspace(0, 1, NB + 1)[1:-1])


def joint(b, n):
    m = np.zeros((n, n))
    if len(b) >= 2:
        np.add.at(m, (b[:-1], b[1:]), 1)
    return m.ravel()


def fingerprint(s, e, r):
    return np.concatenate([joint(np.searchsorted(RB, r, side="right"), 6), joint(np.searchsorted(e_cut, e), NB),
                           joint(np.searchsorted(s_cut, s), NB)])


FP = np.array([fingerprint(*q) for q in seqs])  # raw pair counts: 36 + 25 + 25
BLK = [slice(0, 36), slice(36, 61), slice(61, 86)]


def normed(C):
    out = C.astype(float).copy()
    for b in BLK:
        out[:, b] /= np.maximum(out[:, b].sum(1, keepdims=True), 1)
    return out


def js(p, q):
    m = (p + q) / 2
    kl = lambda a, b: np.where(a > 0, a * np.log(np.where(a > 0, a, 1) / np.where(b > 0, b, 1)), 0).sum(-1)
    return np.sqrt(np.clip((kl(p, m) + kl(q, m)) / 2, 0, None))


FN = normed(FP)
o_lr, o_nn = np.zeros((len(ids), len(classes))), np.zeros((len(ids), len(classes)))
for k in range(4):
    tr = fold_all != k
    sc = StandardScaler().fit(np.sqrt(FN[tr]))
    lr = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(np.sqrt(FN[tr])), y_all[tr])
    o_lr[fold == k] = np.log(np.clip(lr.predict_proba(sc.transform(np.sqrt(FN[pos[fold == k]]))), 1e-12, 1))
    ref = normed(np.array([FP[tr & (y_all == c)].sum(0) for c in range(len(classes))]))  # one fingerprint per LLM
    te = FN[pos[fold == k]]
    dist = sum(js(te[:, None, b], ref[None, :, b]) for b in BLK)
    o_nn[fold == k] = log_softmax(-20 * dist)
M["trace_lr"], M["trace_nn"] = o_lr, o_nn
say(f"trace done {time.time() - T0:.0f}s")

groups = np.full(len(ids), -1)
for k in range(4):
    groups[fold == k] = find_groups(text[fold == k]) + 10_000 * k


def post(z, g):
    z = log_softmax(z)
    out = z.copy()
    for u in np.unique(g):
        m = g == u
        if m.sum() >= 2:
            out[m] -= 0.5 * z[m].mean(0)
    out = log_softmax(out)
    return 0.55 * out + 0.45 * sinkhorn(out)


def mix(w, names, rows=slice(None)):
    return sum(wi * M[n][rows] for wi, n in zip(w, names) if wi)


def score(w, names, mask):
    p = post(mix(w, names, mask), groups[mask]).argmax(1)
    return bacc(y[mask], p)


def tune(names, start, mask):
    w = list(start)
    best = score(w, names, mask)
    for _ in range(3):
        moved = False
        for i in range(len(names)):
            for v in np.round(np.arange(0, 1.01, 0.05), 2):
                if v == w[i]:
                    continue
                cand = w[:i] + [v] + w[i + 1:]
                if sum(cand) == 0:
                    continue
                s = score(cand, names, mask)
                if s > best + 1e-9:
                    best, w, moved = s, cand, True
        if not moved:
            break
    return w, best


def nested(names, start):
    """Weights chosen on 3 folds, scored on the 4th; returns pooled held-out predictions and per-fold weights."""
    pred = np.zeros(len(ids), dtype=int)
    W = []
    for k in range(4):
        w, _ = tune(names, start, fold != k)
        m = fold == k
        pred[m] = post(mix(w, names, m), groups[m]).argmax(1)
        W.append(w)
    return bacc(y, pred), [bacc(y[fold == k], pred[fold == k]) for k in range(4)], W


def sparse_eval(w, names, draws=3):
    raw, pp = [], []
    for s in range(draws):
        rng = np.random.default_rng(1000 + s)
        keep = []
        for k in range(4):
            wk = rng.dirichlet(np.ones(len(classes)))
            r = np.flatnonzero(fold == k)
            pr_id = train.prompt_id.to_numpy()[pos[r]]
            for p in np.unique(pr_id):
                rr = r[pr_id == p]
                n = min(len(rr), int(rng.integers(2, 7)))
                keep += list(rng.choice(rr, size=n, replace=False, p=wk[y[rr]] / wk[y[rr]].sum()))
        keep = np.array(sorted(keep))
        g = find_groups(text[keep])
        z = mix(w, names, keep)
        raw.append(bacc(y[keep], z.argmax(1)))
        pp.append(bacc(y[keep], post(z, g).argmax(1)))
    return np.mean(raw), np.mean(pp), len(keep)


say("\nmembers alone (raw argmax, folds 0-3):")
for n in M:
    say(f"  {n:10s} {bacc(y, M[n].argmax(1)):.4f}")


def run(label, names, start):
    nb, nf, W = nested(names, start)
    w_in, s_in = tune(names, start, np.ones(len(ids), bool))
    sr, sp, n = sparse_eval(w_in, names)
    say(f"\n{label}\n  in-sample best {s_in:.4f} with " + ", ".join(f"{a} {b:.2f}" for a, b in zip(names, w_in) if b)
        + f"\n  NESTED (honest) {nb:.4f}  folds " + " ".join(f"{v:.3f}" for v in nf)
        + "\n  per-fold weights " + " | ".join(",".join(f"{v:.2f}" for v in w) for w in W)
        + f"\n  SPARSE (in-sample weights, {n} rows/draw) raw {sr:.4f} post {sp:.4f}")
    return w_in


V7 = ["s4", "s2mix", "svm_v7", "lgb_ext", "atlas_pool"]
v7w = [0.5, 0.3, 0.1, 0.1, 0.15]
say(f"\nv7 as shipped (fixed weights): {score(v7w, V7, np.ones(len(ids), bool)):.4f}  SPARSE raw/post "
    + "{:.4f} / {:.4f}".format(*sparse_eval(v7w, V7)[:2]))
run("v7 members, weights re-tuned (reference, s4 kept)", V7, v7w)
run("Q1: v7 members with t4 in place of s4, weights re-tuned", ["t4"] + V7[1:], v7w)
TO = ["t4", "svm4", "lgb", "atlas", "letters"]
run("Q2a: train-only, no TRACE", TO, [0.5, 0.1, 0.1, 0.15, 0.1])
run("Q2b: train-only + TRACE", TO + ["trace_lr", "trace_nn"], [0.5, 0.1, 0.1, 0.15, 0.1, 0.0, 0.0])
say(f"\nt4 alone + post: {score([1.0], ['t4'], np.ones(len(ids), bool)):.4f}   total {time.time() - T0:.0f}s")
np.savez(P / "trainonly_parts.npz", ids=np.asarray(ids, dtype=str), y=y, fold=fold, **M)
