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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune"))
from prep_data import typography_mask  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--per-class", type=int, default=300)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--maxlen", type=int, default=320)
cli = ap.parse_args()
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

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
lm = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                  bnb_4bit_compute_dtype=torch.bfloat16),
    dtype=torch.bfloat16, device_map={"": 0}).eval()

Q = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9)


def summarise(lp, ent, top):
    s = lp + ent  # actual surprise minus expected surprise (negative = more predictable than the model expected)
    f = [lp.mean(), lp.std(), *np.quantile(lp, Q), ent.mean(), ent.std(), *np.quantile(ent, (0.25, 0.5, 0.75)),
         s.mean(), s.std(), top.mean(), (lp < -6).mean(), (lp > -0.1).mean()]
    return f


NAMES = (["lp_mean", "lp_std"] + [f"lp_q{int(q * 100)}" for q in Q] + ["ent_mean", "ent_std", "ent_q25", "ent_q50",
         "ent_q75", "sur_mean", "sur_std", "top1", "rare", "certain"])


@torch.no_grad()
def score(texts):
    rows, t0 = [], time.time()
    order = np.argsort([len(t) for t in texts])
    res = [None] * len(texts)
    for b in range(0, len(texts), cli.bs):
        ix = order[b:b + cli.bs]
        enc = tok([tok.bos_token + texts[i] if tok.bos_token else texts[i] for i in ix], return_tensors="pt",
                  padding=True, truncation=True, max_length=cli.maxlen).to(0)
        hid = lm.model(**enc).last_hidden_state  # vocab logits one passage at a time (8 GB card)
        tgt, m = enc.input_ids[:, 1:], enc.attention_mask[:, 1:].bool()
        for j, i in enumerate(ix):
            lg = lm.lm_head(hid[j, :-1][m[j]]).float()
            tj = tgt[j][m[j]]
            lse = torch.logsumexp(lg, -1)  # entropy = lse - sum(p * logit): two vocab-size tensors, not four
            a = (lg.gather(-1, tj[:, None])[:, 0] - lse).cpu().numpy()
            e = (lse - (torch.softmax(lg, -1) * lg).sum(-1)).cpu().numpy()
            t = (lg.argmax(-1) == tj).float().cpu().numpy()
            del lg, lse
            if len(a) < 2:  # empty or one-token passage: neutral row instead of a crash
                a, e, t = np.zeros(2), np.zeros(2), np.zeros(2)
            h = max(len(a) // 2, 4)
            res[i] = summarise(a, e, t) + summarise(a[:h], e[:h], t[:h]) + [np.log(len(a))]
        del hid
        torch.cuda.empty_cache()
        if b % (cli.bs * 100) == 0:
            print(f"  {b}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(res, columns=NAMES + [f"h_{n}" for n in NAMES] + ["log_ntok"])


for name, df in (("host", host), ("pool", pool)):
    F = score(df.text.fillna("").astype(str).tolist())
    F.insert(0, "model", df.model.to_numpy())
    if "id" in df:
        F.insert(0, "id", df.id.astype(str).to_numpy())
    F.to_parquet(OUT / f"atlas_{name}.parquet")
    print(f"{name} done: {F.shape}", flush=True)
