"""Component 5 evaluation: does the base-LM likelihood view (features/atlas.py) help the shipped blend?

Classifier: LightGBM on the ATLAS features, trained per fold k on host rows outside fold k (+ pool rows), scoring
fold k (folds 0-3, never fold 4). Then, as in the public notebook, it only gets a say where the blend is unsure:
  final = blend + w * gate * atlas_logprob,  gate = 1 if blend top1 - top2 < margin (prob space) else 0
compared with the shipped post-processing (centering w0.5 + partial sinkhorn 0.45) applied afterwards.
Writes dashboard entries checkpoints/atlas (ATLAS alone) and checkpoints/blend_atlas (blend + ATLAS best setting).
Run: .venv/Scripts/python.exe features/atlas_eval.py
"""
import json
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
blend = log_softmax(0.5 * d["s4"] + 0.3 * d["s2mix"] + 0.1 * d["svm"] + 0.1 * d["lgb"])  # notebook v6
fold = np.array([fold_of[i] for i in ids])
H = pd.read_parquet(P / "atlas_host.parquet")
H["id"] = H.id.astype(str)
Pl = pd.read_parquet(P / "atlas_pool.parquet")
cols = [c for c in H.columns if c not in ("id", "model")]
Hx = H.set_index("id").loc[ids, cols].to_numpy()
hall_fold = H.id.map(fold_of).to_numpy()
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


PRM = dict(objective="multiclass", num_class=len(classes), learning_rate=0.05, num_leaves=15, min_child_samples=20,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
res = {}
for name, use_pool in (("host", False), ("host+pool", True)):
    oof = np.zeros_like(blend)
    for k in range(4):
        tr = hall_fold != k
        X, Y = H.loc[tr, cols].to_numpy(), H.model[tr].map(cid).to_numpy()
        if use_pool:
            X, Y = np.vstack([X, Pl[cols].to_numpy()]), np.concatenate([Y, Pl.model.map(cid).to_numpy()])
        mdl = lgbm.train(PRM, lgbm.Dataset(X, Y), num_boost_round=300)
        oof[fold == k] = np.log(np.clip(mdl.predict(Hx[fold == k]), 1e-9, 1))
    res[name] = oof
    print(f"ATLAS LightGBM ({name}) alone: {bacc(y, oof.argmax(1)):.4f}", flush=True)

best_name = max(res, key=lambda n: bacc(y, res[n].argmax(1)))
A = res[best_name]
pb = np.exp(blend)
top2 = np.sort(pb, 1)[:, -2:]
margin = top2[:, 1] - top2[:, 0]
base = bacc(y, post(blend).argmax(1))
print(f"\nshipped blend (v6) + centering + partial sinkhorn: {base:.4f}   (argmax {bacc(y, blend.argmax(1)):.4f})")
best = (base, None)
for gate_m in (0.1, 0.2, 0.35, 1.01):
    gate = (margin < gate_m).astype(float)[:, None]
    row = []
    for w in (0.1, 0.2, 0.35, 0.5):
        s = bacc(y, post(blend + w * gate * A).argmax(1))
        row.append(f"w {w:.2f}: {s:.4f}")
        if s > best[0]:
            best = (s, (gate_m, w))
    print(f"gate margin < {gate_m:.2f} ({gate.mean():.0%} of rows): " + "  ".join(row), flush=True)
print(f"best: {best[0]:.4f} at {best[1]} vs {base:.4f}")


def eval_json(lp, label):
    out = {"bal_acc": bacc(y, lp.argmax(1)), "classes": classes, "label": label}
    for key, z in (("", lp), ("_balanced", sinkhorn(lp))):
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        np.add.at(cm, (y, z.argmax(1)), 1)
        out["recall" + key] = [round(float(r), 4) for r in cm.diagonal() / np.maximum(cm.sum(1), 1)]
        out["confusion" + key] = cm.tolist()
    return out


for run, lp, label in (("atlas", A, f"ATLAS likelihood view alone ({best_name}), out-of-fold folds 0-3"),
                       ("blend_atlas", post(blend + (best[1][1] * (margin < best[1][0])[:, None] * A if best[1] else 0)),
                        f"v6 blend + ATLAS {best[1]} + centering + partial sinkhorn, OOF folds 0-3 ({best[0]:.3f} vs {base:.3f})")):
    D = ROOT / "checkpoints" / run
    D.mkdir(exist_ok=True)
    (D / "classes.json").write_text(json.dumps(classes))
    (D / "final_eval.json").write_text(json.dumps(eval_json(lp, label)))
print("dashboard entries written: atlas, blend_atlas")
