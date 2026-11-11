"""Does the 8B stage-2 family (checkpoints/k8f0-3) add to the shipped v7 blend? Folds 0-3 out of fold, never fold 4.

v7 = 0.5 s4 + 0.3 s2mix + 0.1 SVM + 0.1 LightGBM + 0.15 ATLAS (host+pool, class-balanced, 5-seed average, as
features/atlas_fit.py ships it), then prompt centering w 0.5 and partial sinkhorn 0.45 (same post as the notebook).
Keep the 8B only if the best weight beats v7 by at least +0.5 pts and most folds go up.
Run: .venv/Scripts/python.exe research/blend_8b.py
"""
import sys
from pathlib import Path

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from assign import find_groups, log_softmax, sinkhorn  # noqa: E402

P = ROOT / "data/processed"
train = pd.read_csv(ROOT / "data/train.csv")
train["id"] = train.id.astype(str)
classes = sorted(train.model.unique())
cid = {c: i for i, c in enumerate(classes)}
fold_of = {i: k for k, (_, va) in enumerate(GroupKFold(5).split(train, train.model, train.prompt_id))
           for i in train.id.iloc[va]}
d = np.load(P / "blend_parts.npz", allow_pickle=True)
ids, y = [str(i) for i in d["ids"]], d["y"]
fold = np.array([fold_of[i] for i in ids])
assert set(fold) == {0, 1, 2, 3}

# 8B family, aligned to the blend rows
parts = []
for k in range(4):
    r = ROOT / f"checkpoints/k8f{k}"
    vi = pd.read_csv(r / "val_ids.csv").id.astype(str)
    parts.append(pd.DataFrame(log_softmax(np.load(r / "val_logits.npy")), index=vi))
k8 = pd.concat(parts)
assert k8.index.is_unique and set(k8.index) == set(ids), "8B fold rows do not match the blend rows"
k8 = k8.loc[ids].to_numpy()

# ATLAS out of fold, as shipped: host + pool, class-balanced, 5 seeds averaged
H = pd.read_parquet(P / "atlas_host_fp16.parquet")
H["id"] = H.id.astype(str)
Pl = pd.read_parquet(P / "atlas_pool_fp16.parquet")
cols = [c for c in H.columns if c not in ("id", "model")]
Hx = H.set_index("id").loc[ids, cols].to_numpy()
hall_fold = H.id.map(fold_of).to_numpy()
PRM = dict(objective="multiclass", num_class=len(classes), learning_rate=0.05, num_leaves=15, min_child_samples=20,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
           num_threads=6)  # leave CPU for the user's other programs
atlas = np.zeros((len(ids), len(classes)))
for k in range(4):
    tr = hall_fold != k
    X = np.vstack([H.loc[tr, cols].to_numpy(), Pl[cols].to_numpy()])
    Y = np.concatenate([H.model[tr].map(cid).to_numpy(), Pl.model.map(cid).to_numpy()])
    w = (len(Y) / (len(classes) * np.bincount(Y, minlength=len(classes))))[Y]
    pr = np.mean([lgbm.train(dict(PRM, seed=s), lgbm.Dataset(X, Y, weight=w), num_boost_round=300)
                  .predict(Hx[fold == k]) for s in range(5)], axis=0)
    atlas[fold == k] = np.log(np.clip(pr, 1e-9, 1))
    print(f"ATLAS fold {k} done", flush=True)

text = train.set_index("id").loc[ids, "text"].fillna("").to_numpy()
groups = np.full(len(y), -1)
for k in range(4):
    m = fold == k
    groups[m] = find_groups(text[m]) + 10_000 * k


def post(z):
    z = log_softmax(z)
    out = z.copy()
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= 2:
            out[m] -= 0.5 * z[m].mean(0)
    out = log_softmax(out)
    return 0.55 * out + 0.45 * sinkhorn(out)


def score(z):
    p = post(z).argmax(1)
    return bacc(y, p), [bacc(y[fold == k], p[fold == k]) for k in range(4)]


rest = 0.3 * d["s2mix"] + 0.1 * d["svm"] + 0.1 * d["lgb"] + 0.15 * atlas
print(f"\n8B family alone: raw {bacc(y, k8.argmax(1)):.4f} | s4 alone {bacc(y, d['s4'].argmax(1)):.4f}")
base, base_f = score(0.5 * d["s4"] + rest)
print(f"v7 shipped: {base:.4f}  folds " + " ".join(f"{v:.3f}" for v in base_f))
print("w_s4  w_8B    score   delta   folds (delta per fold)")
for w4, w8 in [(0.5, 0.1), (0.5, 0.2), (0.5, 0.3), (0.5, 0.5), (0.4, 0.2), (0.3, 0.3), (0.25, 0.25), (0.35, 0.35)]:
    s, f = score(w4 * d["s4"] + w8 * k8 + rest)
    up = sum(a > b for a, b in zip(f, base_f))
    print(f"{w4:.2f}  {w8:.2f}   {s:.4f}  {100 * (s - base):+.2f}   "
          + " ".join(f"{100 * (a - b):+.1f}" for a, b in zip(f, base_f)) + f"   ({up}/4 folds up)")
