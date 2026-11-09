"""Component 1: partial Sinkhorn. Move predictions only part of the way towards 'every model gets 1/18 of the test'.

final = (1 - alpha) * blend + alpha * sinkhorn(blend), in log-probability space. alpha 0 = plain argmax, 1 = full.
Out-of-fold on folds 0-3. Caution: these rows are exactly balanced (150 per model), which favours alpha = 1 by
construction; partial balancing is a hedge for a test set whose balance we do not know, so the LB decides.
Also reported on a deliberately unbalanced resample (classes kept at 40-100%), to show the hedge at work.
Run: .venv/Scripts/python.exe research/partial_sinkhorn.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score as bacc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from assign import log_softmax, sinkhorn  # noqa: E402

d = np.load(ROOT / "data/processed/blend_parts.npz", allow_pickle=True)
y = d["y"]
blend = log_softmax(0.8 * d["s2mix"] + 0.1 * d["svm"] + 0.1 * d["lgb"])
alphas = (0, 0.25, 0.45, 0.65, 0.85, 1.0)


def scores(rows):
    full = sinkhorn(blend[rows])
    return [bacc(y[rows], ((1 - a) * blend[rows] + a * full).argmax(1)) for a in alphas]


print("alpha:              " + "  ".join(f"{a:5.2f}" for a in alphas))
print("balanced rows:      " + "  ".join(f"{s:.3f}" for s in scores(np.arange(len(y)))))
res = []
for seed in range(5):
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(y), bool)
    for c in np.unique(y):
        r = np.flatnonzero(y == c)
        keep[rng.choice(r, size=int(len(r) * rng.uniform(0.4, 1.0)), replace=False)] = True
    res.append(scores(np.flatnonzero(keep)))
print("unbalanced (5 runs):" + "  ".join(f"{s:.3f}" for s in np.mean(res, 0)))
