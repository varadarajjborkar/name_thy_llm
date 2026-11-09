"""ATLAS-style likelihood features, shared by features/atlas.py (training rows) and the Kaggle notebook (test rows).

One module so the notebook computes exactly what the classifier was trained on. Base model: Qwen3-4B base, 4-bit
NF4 (no double quantisation), fp16 compute (the Kaggle T4 dtype). Per passage, per token: log-prob of the actual
token, entropy of the predicted distribution, whether it was the argmax. Summarised over the whole passage and its
first half (37 numbers).
"""
import time

import numpy as np
import pandas as pd
import torch

Q = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9)
NAMES = (["lp_mean", "lp_std"] + [f"lp_q{int(q * 100)}" for q in Q] + ["ent_mean", "ent_std", "ent_q25", "ent_q50",
         "ent_q75", "sur_mean", "sur_std", "top1", "rare", "certain"])
COLS = NAMES + [f"h_{n}" for n in NAMES] + ["log_ntok"]


def load_lm(model_dir, dtype=torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    lm = AutoModelForCausalLM.from_pretrained(
        model_dir, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                          bnb_4bit_compute_dtype=dtype),
        dtype=dtype, device_map={"": 0}).eval()
    return tok, lm


def summarise(lp, ent, top):
    s = lp + ent  # actual surprise minus expected surprise (negative = more predictable than the model expected)
    return [lp.mean(), lp.std(), *np.quantile(lp, Q), ent.mean(), ent.std(), *np.quantile(ent, (0.25, 0.5, 0.75)),
            s.mean(), s.std(), top.mean(), (lp < -6).mean(), (lp > -0.1).mean()]


@torch.no_grad()
def score(texts, tok, lm, bs=4, maxlen=320, log=print):
    t0 = time.time()
    order = np.argsort([len(t) for t in texts])
    res = [None] * len(texts)
    for b in range(0, len(texts), bs):
        ix = order[b:b + bs]
        enc = tok([tok.bos_token + texts[i] if tok.bos_token else texts[i] for i in ix], return_tensors="pt",
                  padding=True, truncation=True, max_length=maxlen).to(0)
        hid = lm.model(**enc, use_cache=False).last_hidden_state  # vocab logits one passage at a time (memory)
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
        if log and b % (bs * 100) == 0:
            log(f"  atlas {b}/{len(texts)} {time.time() - t0:.0f}s")
    return pd.DataFrame(res, columns=COLS)
