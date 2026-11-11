"""Compute next-word signal features (features/signal_feats.py) for every train.csv row with Qwen3-4B base.
Output (gitignored): data/processed/signal_train.parquet (id, model, SIG_NAMES) and signal_train_seqs.npz (the raw
per-token sequences, for later experiments). About 15 minutes on the laptop GPU.
Run: .venv/Scripts/python.exe features/signal_run.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "features"))
from atlas_feats import load_lm  # noqa: E402
from signal_feats import featurize_signal, token_sequences  # noqa: E402

t0 = time.time()
train = pd.read_csv(ROOT / "data/train.csv")
texts = train.text.fillna("").astype(str).tolist()
tok, lm = load_lm(str(ROOT / "models/qwen3-4b-base"), dtype=torch.float16)  # what the Kaggle T4 notebook uses
print(f"model loaded {time.time() - t0:.0f}s", flush=True)
seqs = token_sequences(texts, tok, lm, log=lambda m: print(m, flush=True))
F = featurize_signal(seqs)
assert np.isfinite(F.to_numpy()).all(), "non-finite signal feature"
out = pd.concat([train[["id", "model"]].astype({"id": str}), F], axis=1)
out.to_parquet(ROOT / "data/processed/signal_train.parquet", index=False)
np.savez_compressed(ROOT / "data/processed/signal_train_seqs.npz", ids=train.id.astype(str).to_numpy(),
                    **{f"{k}_{i}": q[j] for i, q in enumerate(seqs) for j, k in enumerate("ser")})
print(f"done: {out.shape} in {time.time() - t0:.0f}s", flush=True)
