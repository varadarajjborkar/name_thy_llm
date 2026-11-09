"""Final ATLAS classifiers for the notebook: LightGBM on the fp16 likelihood features of ALL train.csv rows plus the
pool rows, class-balanced, 5 seeds (averaged in the notebook). Same parameters as features/atlas_eval.py, whose
seed-averaged OOF gain at weight 0.15 is +1.08 (folds 0-3).
Writes data/kaggle_upload/ntl-lgbm-style/atlas_lgb_s{seed}.txt, atlas_meta.json and a copy of atlas_feats.py.
Run: .venv/Scripts/python.exe features/atlas_fit.py
"""
import json
import shutil
import sys
from pathlib import Path

import lightgbm as lgbm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "features"))
from atlas_feats import COLS  # noqa: E402

P, OUT = ROOT / "data/processed", ROOT / "data/kaggle_upload/ntl-lgbm-style"
classes = sorted(pd.read_csv(ROOT / "data/train.csv").model.unique())
cid = {c: i for i, c in enumerate(classes)}
H, Pl = pd.read_parquet(P / "atlas_host_fp16.parquet"), pd.read_parquet(P / "atlas_pool_fp16.parquet")
X = np.vstack([H[COLS].to_numpy(), Pl[COLS].to_numpy()])
Y = np.concatenate([H.model.map(cid).to_numpy(), Pl.model.map(cid).to_numpy()])
assert np.isfinite(X).all() and len(np.unique(Y)) == len(classes)
cnt = np.bincount(Y, minlength=len(classes))
w = (len(Y) / (len(classes) * cnt))[Y]
PRM = dict(objective="multiclass", num_class=len(classes), learning_rate=0.05, num_leaves=15, min_child_samples=20,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
for seed in range(5):
    m = lgbm.train(dict(PRM, seed=seed), lgbm.Dataset(X, Y, weight=w), num_boost_round=300)
    m.save_model(str(OUT / f"atlas_lgb_s{seed}.txt"))
(OUT / "atlas_meta.json").write_text(json.dumps({"cols": COLS, "classes": classes, "seeds": 5, "maxlen": 320,
                                                 "dtype": "fp16", "rows": int(len(Y))}))
shutil.copy(ROOT / "features/atlas_feats.py", OUT / "atlas_feats.py")
print(f"trained 5 seeds on {len(Y)} rows ({len(H)} host + {len(Pl)} pool); wrote {OUT}")
