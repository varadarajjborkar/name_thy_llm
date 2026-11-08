"""Choose blend weights for Qwen checkpoints + TF-IDF SVM + LightGBM on train.csv rows outside the lockbox.

All inputs are unseen by their models: Qwen stage 1 never trained on train.csv, the SVM scores are out-of-fold
(grouped by prompt), LightGBM was trained on external data only. Small, pre-declared grid; prints every point.
Run: .venv/Scripts/python.exe research/fit_blend.py checkpoints/run2/pred/step1300.npy ...
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as bacc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from assign import log_softmax, sinkhorn  # noqa: E402

ids = pd.read_csv(ROOT / "checkpoints/run2/pred/val_ids.csv")
train = pd.read_csv(ROOT / "data/train.csv")
pos = pd.Series(np.arange(len(train)), index=train.id.astype(str)).loc[ids.id.astype(str)].to_numpy()
classes = sorted(train.model.unique())
y = ids.model.map({c: i for i, c in enumerate(classes)}).to_numpy()
svm = log_softmax(3 * np.load(ROOT / "data/processed/svm_oof.npy")[pos])
lgb = np.log(np.clip(np.load(ROOT / "data/processed/lgbm_ext_val_probs.npy")[pos], 1e-9, 1))
qs = [log_softmax(np.load(f)) for f in sys.argv[1:]]
for f, q in zip(sys.argv[1:], qs):
    print(f"{Path(f).stem:14s} raw {bacc(y, q.argmax(1)):.4f}  sinkhorn {bacc(y, sinkhorn(q).argmax(1)):.4f}")
q = sum(qs) / len(qs)
print(f"{'qwen ensemble':14s} raw {bacc(y, q.argmax(1)):.4f}  sinkhorn {bacc(y, sinkhorn(q).argmax(1)):.4f}")
print("\nw_qwen  w_svm  w_lgbm   raw     sinkhorn")
for wq in (1.0, 0.8, 0.7, 0.6, 0.5):
    for share in (0.5, 0.3):  # share of the rest that goes to the SVM
        ws, wl = (1 - wq) * share, (1 - wq) * (1 - share)
        b = wq * q + ws * svm + wl * lgb
        print(f"{wq:.2f}    {ws:.2f}   {wl:.2f}    {bacc(y, b.argmax(1)):.4f}  {bacc(y, sinkhorn(b).argmax(1)):.4f}")
