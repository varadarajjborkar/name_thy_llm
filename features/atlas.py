"""Component 5 (ATLAS-style): how each passage looks to the plain base LM (Qwen3-4B base, no fine-tuning).

Per passage, per token: log-prob of the actual token, entropy of the predicted distribution, whether it was the
argmax token. Summarised into ~20 numbers (mean, spread, quantiles, surprise minus expected surprise, top-1 rate),
separately for the first half and the whole passage. Our classifiers never see these likelihood statistics.
Rows: all train.csv rows + pool rows (--per-class each, unseen by any of our runs).
Output (gitignored): data/processed/atlas_{host,pool}.parquet (id or row, model, features).
Run: .venv/Scripts/python.exe features/atlas.py --per-class 300
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune"))
from prep_data import typography_mask  # noqa: E402
sys.path.insert(0, str(ROOT / "features"))
from atlas_feats import load_lm, score  # noqa: E402  (shared with the Kaggle notebook)

ap = argparse.ArgumentParser()
ap.add_argument("--per-class", type=int, default=300)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--maxlen", type=int, default=320)
ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])  # fp16 = what the Kaggle T4 notebook uses
cli = ap.parse_args()
DT = torch.float16 if cli.dtype == "fp16" else torch.bfloat16
OUT = ROOT / "data/processed"
MODEL = ROOT / "models/qwen3-4b-base"

host = pd.read_csv(ROOT / "data/train.csv")
used = set()
for f in ("pool_typo_60k", "pool_cont_30k", "pool_cont2"):
    p = OUT / f"{f}.parquet"
    if p.exists():
        used |= set(pd.read_parquet(p, columns=["text"]).text)
pool = pd.read_parquet(OUT / "reserve_typo.parquet", columns=["text", "model"])
pool = pool[typography_mask(pool) & ~pool.text.isin(used) & ~pool.text.isin(set(host.text))]
pool = pd.concat([g.sample(n=min(len(g), cli.per_class), random_state=11)
                  for _, g in pool.groupby("model")]).reset_index(drop=True)
print(f"host {len(host)}, pool {len(pool)} ({pool.model.value_counts().min()}-{pool.model.value_counts().max()} per class)",
      flush=True)

tok, lm = load_lm(MODEL, DT)


for name, df in (("host", host), ("pool", pool)):
    F = score(df.text.fillna("").astype(str).tolist(), tok, lm, cli.bs, cli.maxlen)
    F.insert(0, "model", df.model.to_numpy())
    if "id" in df:
        F.insert(0, "id", df.id.astype(str).to_numpy())
    F.to_parquet(OUT / f"atlas_{name}_{cli.dtype}.parquet")
    print(f"{name} done: {F.shape}", flush=True)
