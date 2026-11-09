"""Blend weights over several stage-2 families plus stage 1, SVM and LightGBM, on folds 0-3 OOF (never fold 4).

Each family is a set of CV runs, checkpoints/<prefix>f{k} for k = 0..3 (s2 is the run2-based family, named s2f0/s2f1
from research/runs/e005 for folds 0-1). Every member is unseen on the rows it scores.
Run: .venv/Scripts/python.exe research/fit_blend_multi.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as bacc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from assign import log_softmax, sinkhorn  # noqa: E402

FAMILIES = {  # name -> fold dirs 0..3
    "s2": [ROOT / "research/runs/e005/f0", ROOT / "research/runs/e005/f1", ROOT / "checkpoints/s2f2",
           ROOT / "checkpoints/s2f3"],
    "s3": [ROOT / f"checkpoints/s3f{k}" for k in range(4)],
    "h3": [ROOT / f"checkpoints/h3f{k}" for k in range(4)],
}
train = pd.read_csv(ROOT / "data/train.csv")
train["id"] = train.id.astype(str)
classes = sorted(train.model.unique())


def family(dirs):
    parts = []
    for d in dirs:
        ids = pd.read_csv(d / "val_ids.csv").id.astype(str)
        parts.append(pd.DataFrame(log_softmax(np.load(d / "val_logits.npy")), index=ids))
    return pd.concat(parts)


F = {k: family(v) for k, v in FAMILIES.items()}
ids = F["s2"].index
F = {k: v.loc[ids].to_numpy() for k, v in F.items()}
y = train.set_index("id").loc[ids, "model"].map({c: i for i, c in enumerate(classes)}).to_numpy()
pos = pd.Series(np.arange(len(train)), index=train.id).loc[ids].to_numpy()
p1 = pd.read_csv(ROOT / "checkpoints/run2/pred/val_ids.csv").id.astype(str)
r1 = pd.Series(np.arange(len(p1)), index=p1).loc[ids].to_numpy()
s1 = (log_softmax(np.load(ROOT / "checkpoints/run2/pred/final.npy"))[r1]
      + log_softmax(np.load(ROOT / "checkpoints/k_host3/val_logits.npy"))[r1]) / 2  # same row order as run2 pred
svm = log_softmax(3 * np.load(ROOT / "data/processed/svm_oof.npy")[pos])
lgb = np.log(np.clip(np.load(ROOT / "data/processed/lgbm_ext_val_probs.npy")[pos], 1e-9, 1))

print(f"rows {len(y)}")
for k, v in F.items():
    print(f"{k}: raw {bacc(y, v.argmax(1)):.4f}  sinkhorn {bacc(y, sinkhorn(v).argmax(1)):.4f}")
s2mix = sum(F.values()) / len(F)
print(f"mean of families: raw {bacc(y, s2mix.argmax(1)):.4f}  sinkhorn {bacc(y, sinkhorn(s2mix).argmax(1)):.4f}")


def eval_json(lp, label):
    """Recall and confusion, raw and class-balanced, in the trainer's final_eval.json format (for the dashboard)."""
    out = {"bal_acc": bacc(y_, lp.argmax(1)), "classes": classes, "label": label}
    for key, z in (("", lp), ("_balanced", sinkhorn(lp))):
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        np.add.at(cm, (y_, z.argmax(1)), 1)
        out["recall" + key] = [round(float(r), 4) for r in cm.diagonal() / np.maximum(cm.sum(1), 1)]
        out["confusion" + key] = cm.tolist()
    return out


import json  # noqa: E402

for name, dirs in FAMILIES.items():  # Kaggle-trained folds came back without final_eval.json
    for d in dirs:
        if not (d / "final_eval.json").exists() and d.parent.name == "checkpoints":
            ids_d = pd.read_csv(d / "val_ids.csv")
            y_ = ids_d.model.map({c: i for i, c in enumerate(classes)}).to_numpy()
            (d / "final_eval.json").write_text(json.dumps(eval_json(log_softmax(np.load(d / "val_logits.npy")),
                                                                     f"fold validation ({len(y_)} passages)")))
y_ = y
CHOSEN = 0.8 * s2mix + 0.1 * svm + 0.1 * lgb  # the shipped blend (kernel_qwen v4)
np.savez(ROOT / "data/processed/blend_oof.npz", ids=np.asarray(ids), lp=log_softmax(CHOSEN), y=y)  # for challengers
B = ROOT / "checkpoints/blend_best"
B.mkdir(exist_ok=True)
(B / "classes.json").write_text(json.dumps(classes))
(B / "final_eval.json").write_text(json.dumps(eval_json(
    CHOSEN, f"best blend, out-of-fold on folds 0-3 ({len(y):,} passages; LB 0.555)")))

print("\nw_stage2(equal over families)  w_s1  w_svm  w_lgbm   raw     sinkhorn")
for w2, w1, wx in [(1, 0, 0), (0.8, 0, 0.1), (0.7, 0.1, 0.1), (0.6, 0.2, 0.1), (0.6, 0.1, 0.15), (0.5, 0.2, 0.15)]:
    b = w2 * s2mix + w1 * s1 + wx * svm + wx * lgb
    print(f"{w2:.2f}                          {w1:.2f}  {wx:.2f}   {wx:.2f}    {bacc(y, b.argmax(1)):.4f}  "
          f"{bacc(y, sinkhorn(b).argmax(1)):.4f}")
