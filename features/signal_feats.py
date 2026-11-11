"""Next-word probability as a SIGNAL (the user's signal-processing idea, 2026-11-11).

A plain base LM (Qwen3-4B base, 4-bit NF4, fp16 compute, same loading as features/atlas_feats.py and the Kaggle
notebook) reads each text once. For every token it records
  s  surprise of the actual token        = -log p(token | the tokens before)
  e  expected surprise (entropy) of the model's prediction at that point
  r  rank of the actual token among the model's guesses (0 = the model's top guess)
and the text becomes three sequences. No word identities are used, only how predictable each choice was.

Features (SIG_NAMES, 30 numbers per text):
  GLTR rank buckets   share of tokens with rank 0, 1-2, 3-9, 10-99, 100-999, 1000+ (Gehrmann et al. 2019)
  spectrum            FFT power of the standardised surprise sequence in 6 frequency bands, spectral centroid and
                      flatness (FACE, Yang et al. 2023: Fourier analysis of cross-entropy): the rhythm of
                      predictable and surprising words
  autocorrelation     surprise at lags 1, 2, 3, 5; entropy at lag 1; excess surprise (s - e) at lag 1
  runs                mean and max run of top-1 tokens; P(top-1 | previous top-1); P(rank >= 10 | previous >= 10)
  shape               excess surprise mean and std; burstiness (std of an 8-token rolling mean); trend (slope)
"""
import time

import numpy as np
import pandas as pd
import torch

BANDS = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
SIG_NAMES = (["rank0", "rank1_2", "rank3_9", "rank10_99", "rank100_999", "rank1000p"]
             + [f"pow_{lo}_{hi}" for lo, hi in BANDS] + ["spec_centroid", "spec_flatness"]
             + ["ac_s1", "ac_s2", "ac_s3", "ac_s5", "ac_e1", "ac_d1"]
             + ["run0_mean", "run0_max", "p00", "p_hi_hi"]
             + ["d_mean", "d_std", "burst", "trend", "s_mean", "log_ntok"])


def _ac(x, lag):
    if len(x) <= lag + 1 or x.std() < 1e-9:
        return 0.0
    a, b = x[:-lag], x[lag:]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _runs(flag):
    runs, cur = [], 0
    for f in flag:
        if f:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def signal_features(s, e, r):
    """30 numbers from one text's sequences (numpy arrays of equal length)."""
    n = len(s)
    if n < 4:
        return [0.0] * len(SIG_NAMES[:-1]) + [float(np.log(max(n, 1)))]
    buckets = [(r == 0).mean(), ((r >= 1) & (r <= 2)).mean(), ((r >= 3) & (r <= 9)).mean(),
               ((r >= 10) & (r <= 99)).mean(), ((r >= 100) & (r <= 999)).mean(), (r >= 1000).mean()]
    z = (s - s.mean()) / (s.std() + 1e-9)
    pw = np.abs(np.fft.rfft(z)) ** 2
    fr = np.fft.rfftfreq(n)
    pw, fr = pw[1:], fr[1:]  # drop the zero frequency (the mean, removed anyway)
    tot = pw.sum() + 1e-12
    bands = [pw[(fr >= lo) & (fr < hi if hi < 0.5 else fr <= hi)].sum() / tot for lo, hi in BANDS]
    centroid = float((fr * pw).sum() / tot)
    flat = float(np.exp(np.mean(np.log(pw + 1e-12))) / (pw.mean() + 1e-12))
    d = s - e
    top = r == 0
    hi = r >= 10
    runs = _runs(top)
    p00 = float((top[1:] & top[:-1]).sum() / max(top[:-1].sum(), 1))
    phh = float((hi[1:] & hi[:-1]).sum() / max(hi[:-1].sum(), 1))
    roll = np.convolve(s, np.ones(8) / 8, mode="valid") if n >= 8 else s
    trend = float(np.polyfit(np.arange(n) / n, s, 1)[0])
    return ([float(b) for b in buckets] + [float(b) for b in bands] + [centroid, flat]
            + [_ac(s, 1), _ac(s, 2), _ac(s, 3), _ac(s, 5), _ac(e, 1), _ac(d, 1)]
            + [float(np.mean(runs)) if runs else 0.0, float(max(runs)) if runs else 0.0, p00, phh]
            + [float(d.mean()), float(d.std()), float(roll.std()), trend, float(s.mean()), float(np.log(n))])


@torch.no_grad()
def token_sequences(texts, tok, lm, bs=4, maxlen=320, log=print):
    """Per text: (s, e, r) arrays, same tokenisation as features/atlas_feats.score (BOS + text, max 320 tokens)."""
    t0 = time.time()
    order = np.argsort([len(t) for t in texts])
    out = [None] * len(texts)
    for b in range(0, len(texts), bs):
        ix = order[b:b + bs]
        enc = tok([tok.bos_token + texts[i] if tok.bos_token else texts[i] for i in ix], return_tensors="pt",
                  padding=True, truncation=True, max_length=maxlen).to(0)
        if enc.input_ids.shape[1] < 2:
            for i in ix:
                out[i] = (np.zeros(0), np.zeros(0), np.zeros(0))
            continue
        hid = lm.model(**enc, use_cache=False).last_hidden_state
        tgt, m = enc.input_ids[:, 1:], enc.attention_mask[:, 1:].bool()
        for j, i in enumerate(ix):
            lg = lm.lm_head(hid[j, :-1][m[j]]).float()
            tj = tgt[j][m[j]]
            act = lg.gather(-1, tj[:, None])
            lse = torch.logsumexp(lg, -1)
            s = (lse - act[:, 0]).cpu().numpy()
            e = (lse - (torch.softmax(lg, -1) * lg).sum(-1)).cpu().numpy()
            r = (lg > act).sum(-1).cpu().numpy()
            out[i] = (s.astype(np.float32), e.astype(np.float32), r.astype(np.int32))
            del lg, act, lse
        del hid
        torch.cuda.empty_cache()
        if log and b % (bs * 100) == 0:
            log(f"  signal {b}/{len(texts)} {time.time() - t0:.0f}s")
    return out


def featurize_signal(seqs):
    return pd.DataFrame([signal_features(*q) for q in seqs], columns=SIG_NAMES)
