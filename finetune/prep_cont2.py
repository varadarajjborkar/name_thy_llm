"""Second stage-1 continuation set: the rest of the typography-filtered pool that run2 and run3 never saw.

Up to 6,000 unseen rows per model (big classes capped so they do not swamp the rest). qwen3.8-max and
gemini-3.7-flash have no unseen rows left, so 3,000 already-seen rows each are replayed to avoid forgetting them.
Output: data/processed/pool_cont2.parquet (about 92.5k rows).
Run: .venv/Scripts/python.exe finetune/prep_cont2.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_data import typography_mask  # noqa: E402

seen = pd.concat([pd.read_parquet(P / "pool_typo_60k.parquet", columns=["text", "model"]),
                  pd.read_parquet(P / "pool_cont_30k.parquet", columns=["text", "model"])])
r = pd.read_parquet(P / "reserve_typo.parquet", columns=["text", "model"])
r = r[typography_mask(r) & ~r.text.isin(set(seen.text))]
assert not set(r.text) & set(pd.read_csv(ROOT / "data/train.csv").text), "host passage in the pool"
new = pd.concat([g.sample(n=min(len(g), 6000), random_state=4) for _, g in r.groupby("model")])
replay = pd.concat([seen[seen.model == m].sample(n=3000, random_state=4)
                    for m in ("qwen3.8-max", "gemini-3.7-flash")])
out = pd.concat([new, replay]).sample(frac=1, random_state=4)
out[["text", "model"]].to_parquet(P / "pool_cont2.parquet", index=False)
print(f"written {len(out):,} rows ({len(new):,} unseen + {len(replay):,} replay)")
print(out.model.value_counts().to_string())
