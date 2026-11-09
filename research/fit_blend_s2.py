"""Blend weights with stage-2 out-of-fold predictions (folds 0-3, never the lockbox fold 4).

Members on the same rows, all unseen by the model that scored them:
  s2   : stage-2 OOF logits, one CV run per fold (val_logits.npy + val_ids.csv per fold)
  s1   : stage-1 run2 final + step 1700 (never trained on train.csv)
  svm  : TF-IDF SVM OOF (grouped by prompt), lgbm: LightGBM trained on external data
Run: .venv/Scripts/python.exe research/fit_blend_s2.py <fold dir> <fold dir> ...
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as bacc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from assign import log_softmax, sinkhorn  # noqa: E402

train = pd.read_csv(ROOT / "data/train.csv")
train["id"] = train.id.astype(str)
classes = sorted(train.model.unique())
parts = []
for d in sys.argv[1:]:
    ids = pd.read_csv(Path(d) / "val_ids.csv")
    parts.append(pd.DataFrame(log_softmax(np.load(Path(d) / "val_logits.npy")), index=ids.id.astype(str)))
s2 = pd.concat(parts)
assert s2.index.is_unique
ids = s2.index
y = train.set_index("id").loc[ids, "model"].map({c: i for i, c in enumerate(classes)}).to_numpy()
pos = pd.Series(np.arange(len(train)), index=train.id).loc[ids].to_numpy()

p1 = pd.read_csv(ROOT / "checkpoints/run2/pred/val_ids.csv").id.astype(str)
s1_rows = pd.Series(np.arange(len(p1)), index=p1).loc[ids].to_numpy()
s1 = sum(log_softmax(np.load(ROOT / f"checkpoints/run2/pred/{n}.npy"))[s1_rows] for n in ("final", "step1700")) / 2
svm = log_softmax(3 * np.load(ROOT / "data/processed/svm_oof.npy")[pos])
lgb = np.log(np.clip(np.load(ROOT / "data/processed/lgbm_ext_val_probs.npy")[pos], 1e-9, 1))
S2 = s2.to_numpy()
print(f"rows {len(y)} | s2 raw {bacc(y, S2.argmax(1)):.4f} sk {bacc(y, sinkhorn(S2).argmax(1)):.4f} "
      f"| s1 raw {bacc(y, s1.argmax(1)):.4f}")
print("\nw_s2  w_s1  w_svm  w_lgbm   raw     sinkhorn")
for w2, w1, ws, wl in [(1, 0, 0, 0), (0.8, 0, 0.1, 0.1), (0.7, 0.1, 0.1, 0.1), (0.6, 0.2, 0.1, 0.1),
                       (0.5, 0.3, 0.1, 0.1), (0.6, 0.1, 0.15, 0.15), (0.5, 0.2, 0.15, 0.15), (0.7, 0.3, 0, 0)]:
    b = w2 * S2 + w1 * s1 + ws * svm + wl * lgb
    print(f"{w2:.2f}  {w1:.2f}  {ws:.2f}   {wl:.2f}    {bacc(y, b.argmax(1)):.4f}  {bacc(y, sinkhorn(b).argmax(1)):.4f}")
