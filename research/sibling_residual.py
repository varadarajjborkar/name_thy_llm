"""OOF sibling-style residual, rebuilt after the method of the public "Name That LLM | Who am I?" notebook (v40A).

Per model family (label prefix claude / gemini / gpt / glm) a heavily regularised logistic regression on stylometry
(punctuation markers, transition words, function-word rates, sentence and word shape) separates the siblings.
  reliability = clip((inner-OOF balanced accuracy - chance) / 0.15, 0, 1)    (inner 5-fold GroupKFold by prompt)
  gate        = clip((0.65 - top1/top2 sibling margin) / 0.65, 0, 1)         (only disputed rows)
  strength    = max_alpha * reliability * gate                               (max_alpha 0.08 as in the notebook)
  within the family: p_new = (1 - strength) * p_blend + strength * p_style, family total probability unchanged.
Scored out of fold on folds 0-3 (fit on host rows outside fold k, never fold 4 as validation), against our blend,
with and without partial Sinkhorn (alpha 0.45). Host rows only, as in the notebook.
Run: .venv/Scripts/python.exe research/sibling_residual.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from assign import log_softmax, sinkhorn  # noqa: E402

MARKERS = [",", ".", ";", ":", "!", "?", "—", "–", "-", "’", "‘", "“", "”", '"', "'",
           "(", ")", "[", "]", "*", "`", "…", "/", "\n", "\n\n", "\n-", "\n1.", "##", "**", " e.g.", " i.e.",
           "However,", "Overall,", "In conclusion", "Therefore,", "Moreover,", "First,", "Finally,", " - ", ": ",
           "  "]
WORDS = ["the", "and", "but", "that", "which", "with", "into", "from", "for", "you", "your", "our", "we", "i", "it",
         "its", "they", "their", "there", "this", "these", "those", "however", "therefore", "moreover", "although",
         "because", "perhaps", "might", "would", "could", "should", "like", "such", "just", "very", "also", "not",
         "if", "or", "as", "so", "by", "of", "to", "in", "on", "at", "is", "are", "be", "was", "were", "have", "has",
         "had", "may", "must", "all", "any", "each", "every", "many", "some", "more", "most", "only", "even",
         "still", "then", "when", "where", "why", "how", "what", "who"]
WRE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
SRE = re.compile(r"(?<=[.?!])\s+")


def style_row(t):
    s = WRE.findall(t)
    lo = [z.lower() for z in s]
    n, nt = max(len(s), 1), max(len(t), 1)
    lens = np.array([len(z) for z in s] or [0])
    sl = np.array([len(WRE.findall(z)) for z in SRE.split(t) if z.strip()] or [0])
    base = [np.log1p(n), np.log1p(nt), nt / n, len(set(lo)) / n, lens.mean(), lens.std(), sl.mean(), sl.std(),
            sl.max(), np.mean(lens >= 8), sum(c.isupper() for c in t) / nt, sum(c.isdigit() for c in t) / nt,
            sum(ord(c) > 127 for c in t) / nt, sum(c == "\n" for c in t) / nt]
    base += [min(t.count(m) / n, 3) for m in MARKERS]
    cnt = {}
    for w in lo:
        cnt[w] = cnt.get(w, 0) + 1
    return np.array(base + [cnt.get(w, 0) / n for w in WORDS], dtype=np.float32)


FAMILIES = [("claude", 0.003), ("gemini", 0.003), ("gpt", 0.003), ("glm", 0.03)]


def residual(lp, Xtr, ytr, gtr, Xte, classes, max_alpha=0.08):
    out = lp.astype(np.float64).copy()
    info = {}
    for fam, C in FAMILIES:
        cols = np.array([i for i, c in enumerate(classes) if c.startswith(fam)])
        m = np.isin(ytr, cols)
        X, tgt, grp = Xtr[m], np.searchsorted(cols, ytr[m]), gtr[m]
        oof = np.empty(len(tgt), int)
        for a, b in GroupKFold(5).split(X, tgt, grp):
            mdl = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=400, class_weight="balanced"))
            oof[b] = mdl.fit(X[a], tgt[a]).predict(X[b])
        acc, chance = bacc(tgt, oof), 1 / len(cols)
        rel = float(np.clip((acc - chance) / 0.15, 0, 1))
        info[fam] = (round(acc, 3), round(rel, 2))
        if rel <= 0:
            continue
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=400, class_weight="balanced"))
        sp = np.maximum(mdl.fit(X, tgt).predict_proba(Xte), 1e-8)
        old = out[:, cols]
        op = softmax(old, axis=1)
        top2 = np.sort(op, 1)[:, -2:]
        gate = np.clip((0.65 - (top2[:, 1] - top2[:, 0])) / 0.65, 0, 1)
        s = max_alpha * rel * gate
        mix = (1 - s[:, None]) * op + s[:, None] * sp
        out[:, cols] = logsumexp(old, axis=1, keepdims=True) + np.log(np.maximum(mix, 1e-12))
    return out, info


d = np.load(ROOT / "data/processed/blend_parts.npz", allow_pickle=True)
ids, y = [str(i) for i in d["ids"]], d["y"]
blend = log_softmax(0.8 * d["s2mix"] + 0.1 * d["svm"] + 0.1 * d["lgb"])
train = pd.read_csv(ROOT / "data/train.csv")
train["id"] = train.id.astype(str)
classes = sorted(train.model.unique())
cid = {c: i for i, c in enumerate(classes)}
folds = list(GroupKFold(5).split(train, train.model, train.prompt_id))
fold_of = {i: k for k, (_, va) in enumerate(folds) for i in train.id.iloc[va]}
S_all = np.stack([style_row(t) for t in train.text.fillna("").astype(str)])
yall, gall = train.model.map(cid).to_numpy(), train.prompt_id.to_numpy()
pos = pd.Series(np.arange(len(train)), index=train.id).loc[ids].to_numpy()
fold = np.array([fold_of[i] for i in ids])

results = {}
for max_alpha in (0.08, 0.2, 0.5):
    new = blend.copy()
    for k in range(4):
        tr = np.flatnonzero(train.id.map(fold_of).to_numpy() != k)  # host rows outside fold k (fold 4 may train)
        va = np.flatnonzero(fold == k)
        new[va], info = residual(blend[va], S_all[tr], yall[tr], gall[tr], S_all[pos[va]], classes, max_alpha)
        if k == 0 and max_alpha == 0.08:
            print("family (inner OOF acc, reliability) on fold-0 training rows:", info)
    results[max_alpha] = new
print(f"blend:                       argmax {bacc(y, blend.argmax(1)):.4f}  partial-sinkhorn {bacc(y, (0.55 * blend + 0.45 * sinkhorn(blend)).argmax(1)):.4f}")
for a, z in results.items():
    z = log_softmax(z)
    print(f"blend + residual (max {a:.2f}): argmax {bacc(y, z.argmax(1)):.4f}  partial-sinkhorn {bacc(y, (0.55 * z + 0.45 * sinkhorn(z)).argmax(1)):.4f}")
