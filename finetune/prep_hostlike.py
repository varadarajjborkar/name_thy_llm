"""Build the host-faithful extra training set from prvsiyan's extended dataset.

Outputs (data/processed/, gitignored):
  hostlike.parquet         'new' + 'new2' splits (prompts NOT taken from the host train set), filtered:
                           no markdown / refusal / truncation, inside the word range, host typography respected.
                           Kimi rows come from 'new_kimi_nothink' (closer to the host's Kimi typography).
  train_regen_norm.parquet the 150 host train prompts re-answered by all 18 models. FINAL FIT ONLY: those prompts
                           are the validation prompts, so using them in any measured run would inflate the score.

Run: .venv/Scripts/python.exe finetune/prep_hostlike.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_data import typography_mask  # noqa: E402  (same rule as the pool filter)

SRC = ROOT / "data/external/prvsiyans-extended-name-that-llm-dataset/passages.csv"
REGEN = ROOT / "data/external/ntl-train-regen-normalised/train_regen_normalised.parquet"
OUT = ROOT / "data/processed"

p = pd.read_csv(SRC)
p = p[p.split.isin(["new", "new2", "new_kimi_nothink"])]
p = p[~((p.split == "new") & (p.model == "kimi-k3"))]  # replaced by the reasoning-off Kimi rerun
ok = ~p.is_refusal.astype(bool) & ~p.is_truncated.astype(bool) & ~p.is_empty.astype(bool) \
    & ~p.has_markdown.astype(bool) & p.in_word_range.astype(bool)
print(f"prvsiyan new/new2/kimi rerun: {len(p):,} rows, {int(ok.sum()):,} pass quality checks")
p = p[ok].copy()
p = p[typography_mask(p)]
p["prompt_group"] = p.split.str.replace("new_kimi_nothink", "new") + "/" + p.prompt_id.astype(str)
host = pd.read_csv(ROOT / "data/train.csv")
assert set(p.model) <= set(host.model), "unknown label"
assert not set(p.text) & set(host.text), "a host passage leaked into the extra set"
out = p[["text", "model", "prompt_group", "split", "provider_route"]].sample(frac=1, random_state=0).reset_index(drop=True)
out.to_parquet(OUT / "hostlike.parquet", index=False)
print(f"hostlike.parquet: {len(out):,} rows, {out.prompt_group.nunique()} prompts")
print(out.model.value_counts().sort_index().to_string())

r = pd.read_parquet(REGEN)
r = r[r.model.isin(host.model.unique())]
r = r[~r.text.isin(set(host.text))]
r.to_parquet(OUT / "train_regen_norm.parquet", index=False)
print(f"\ntrain_regen_norm.parquet: {len(r):,} rows (final fit only)")
