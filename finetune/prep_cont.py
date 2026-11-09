"""Fresh stage-1 continuation sample from pool rows run2 never saw (reserve_typo.parquet), same typography filter.

At most 1,800 rows per model (qwen3.8-max has only 929 left). Output: data/processed/pool_cont_30k.parquet (29,729 rows).
Run: .venv/Scripts/python.exe finetune/prep_cont.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_data import typography_mask  # noqa: E402

r = pd.read_parquet(ROOT / "data/processed/reserve_typo.parquet")
r = r[typography_mask(r)]
assert not set(r.text) & set(pd.read_csv(ROOT / "data/train.csv").text), "host passage in the pool"
s = pd.concat([g.sample(n=min(len(g), 1800), random_state=1) for _, g in r.groupby("model")]).sample(frac=1, random_state=1)
s[["text", "model"]].to_parquet(ROOT / "data/processed/pool_cont_30k.parquet", index=False)
print("written", len(s))
