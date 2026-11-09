"""Adversarial booster: a second-stage model aimed at the blend's mistakes, cross-fitted over folds 0-3.

Inputs per host row: the blend's out-of-fold log-probabilities (18) + compact frozen stage-1 Qwen features
(PCA fitted on pool rows only, so no host label touches it). Rows the blend gets wrong are up-weighted (alpha),
so the booster spends its capacity on the blend's errors. For each fold k it trains on the other three folds'
rows and predicts fold k, then the result is mixed back with the blend. Never touches fold 4.
Run: .venv/Scripts/python.exe features/booster.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
train = pd.read_csv(ROOT / "data/train.csv")
fold_of = {}
for k, (_, va) in enumerate(GroupKFold(5).split(train, train.model, train.prompt_id)):
    fold_of.update({str(i): k for i in train.id.iloc[va]})

bl = np.load(P / "blend_oof.npz", allow_pickle=True)
ids = [str(i) for i in bl["ids"]]
lp, y = bl["lp"], bl["y"]
hmeta = pd.read_parquet(P / "feats_run3_host.parquet")
hX = np.load(P / "feats_run3_host.npy").astype(np.float32)[
    pd.Series(np.arange(len(hmeta)), index=hmeta.id.astype(str)).loc[ids].to_numpy()]
pX = np.load(P / "feats_run3_pool.npy").astype(np.float32)
fold = np.array([fold_of[i] for i in ids])
sc = StandardScaler().fit(pX)
print(f"rows {len(y)} | blend raw {bacc(y, lp.argmax(1)):.4f}")
for dims in (0, 16, 64):
    F = lp if dims == 0 else np.hstack([lp, PCA(dims, random_state=0).fit(sc.transform(pX)).transform(sc.transform(hX))])
    for alpha in (0.0, 1.0, 3.0):
        out = np.zeros_like(lp)
        for k in range(4):
            tr, va = fold != k, fold == k
            wrong = (lp[tr].argmax(1) != y[tr]).astype(float)
            Fs = StandardScaler().fit(F[tr])
            clf = LogisticRegression(C=0.1, max_iter=3000).fit(Fs.transform(F[tr]), y[tr], sample_weight=1 + alpha * wrong)
            out[va] = clf.predict_log_proba(Fs.transform(F[va]))
        mix = [f"{bacc(y, (w * out + (1 - w) * lp).argmax(1)):.4f}" for w in (1.0, 0.5, 0.3)]
        print(f"pca {dims:3d} alpha {alpha}: booster alone {mix[0]} | mixed 50/50 {mix[1]} | 30/70 {mix[2]}")
