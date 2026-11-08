"""Build the fine-tuning sample from the Name That LLM open pool.

Outputs (data/processed/, gitignored):
  train_100k.parquet   the stratified training sample
  reserve_pool.parquet everything else (unsampled rows + any excluded rows), column `reserve_reason`

Sampling:
  1. Exclude rows with any defect flag (and kb_proxy rows if --drop-kb-proxy).
  2. Equal quota per model (water-filling): small classes give everything they have,
     the leftover quota is shared equally by the larger classes.
  3. Inside a class, spread picks equally across source_dataset (again water-filling),
     then prefer many distinct prompt_group values by shuffling groups first.

Run:  .venv/Scripts/python.exe finetune/prep_data.py [--n 100000] [--seed 42] [--drop-kb-proxy]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data/external/name-that-llm-open-pool/ntl_open_pool.parquet"
OUT = ROOT / "data/processed"
DEFECT_FLAGS = ["flag_gemini37_routing_risk", "flag_near_train_prompt", "flag_train_prompt_regen"]


def water_fill(sizes, total):
    """Split `total` as equally as possible across buckets with capacities `sizes`."""
    sizes = np.asarray(sizes, dtype=np.int64)
    alloc = np.zeros_like(sizes)
    left = int(min(total, sizes.sum()))
    active = np.arange(len(sizes))
    while left > 0 and len(active):
        share = max(left // len(active), 1)
        for i in list(active):
            take = min(share, sizes[i] - alloc[i], left)
            alloc[i] += take
            left -= take
            if alloc[i] >= sizes[i]:
                active = active[active != i]
            if left == 0:
                break
    return alloc


def pick_in_class(g, k, rng):
    """Choose k rows from one class: equal across sources, diverse across prompt groups."""
    srcs = sorted(g.source_dataset.unique())
    caps = [int((g.source_dataset == s).sum()) for s in srcs]
    per_src = water_fill(caps, k)
    chosen = []
    for s, n in zip(srcs, per_src):
        if n == 0:
            continue
        gs = g[g.source_dataset == s].copy()
        # one random row per prompt_group first, then the rest, so groups are spread out
        gs["_r"] = rng.random(len(gs))
        gs["_rank"] = gs.groupby("prompt_group")["_r"].rank(method="first")
        gs = gs.sort_values(["_rank", "_r"]).head(int(n))
        chosen.append(gs.index.to_numpy())
    return np.concatenate(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-kb-proxy", action="store_true")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    d = pd.read_parquet(POOL)
    d["reserve_reason"] = ""

    bad = d[DEFECT_FLAGS].any(axis=1)
    d.loc[bad, "reserve_reason"] = "defect_flag"
    if a.drop_kb_proxy:
        d.loc[d.flag_kb_proxy & ~bad, "reserve_reason"] = "kb_proxy"
    elig = d[d.reserve_reason == ""]
    print(f"pool {len(d):,}  eligible {len(elig):,}  excluded {len(d) - len(elig):,}")

    models = sorted(elig.model.unique())
    quota = water_fill([int((elig.model == m).sum()) for m in models], a.n)
    picked = np.concatenate([pick_in_class(elig[elig.model == m], int(q), rng) for m, q in zip(models, quota)])

    d.loc[picked, "reserve_reason"] = "train"
    d.loc[(d.reserve_reason == ""), "reserve_reason"] = "unsampled"
    train = d[d.reserve_reason == "train"].sample(frac=1, random_state=a.seed).reset_index(drop=True)
    reserve = d[d.reserve_reason != "train"].reset_index(drop=True)

    OUT.mkdir(parents=True, exist_ok=True)
    train.drop(columns=["reserve_reason"]).to_parquet(OUT / "train_100k.parquet", index=False)
    reserve.to_parquet(OUT / "reserve_pool.parquet", index=False)

    print(f"train {len(train):,}   reserve {len(reserve):,}")
    tab = pd.DataFrame({"train": train.model.value_counts(), "reserve": reserve.model.value_counts()}).fillna(0).astype(int)
    tab["srcs_in_train"] = train.groupby("model").source_dataset.nunique()
    tab["prompt_groups"] = train.groupby("model").prompt_group.nunique()
    print(tab.sort_values("train").to_string())
    print("reserve reasons:", reserve.reserve_reason.value_counts().to_dict())
    print("words mean train/pool:", round(train.n_words.mean(), 1), round(d.n_words.mean(), 1))


if __name__ == "__main__":
    main()
