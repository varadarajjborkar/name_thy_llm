"""Per-model bigram language models on content words (forward + backward).

Stage 1: score every text under each model's forward P(next|prev) and backward
         P(prev|next) bigram LM; predict argmax of  fwd + w * bwd.
Stage 2: logistic regression on the out-of-fold per-model scores.
All evaluation is GroupKFold by prompt_id (each prompt is answered by all models).
"""
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

DATA = "/Users/varadarajborkar/Desktop/call_llm/data"
SENT_SPLIT = re.compile(r"[.!?;:\n]+")
WORD = re.compile(r"[a-z][a-z'\-]*")
LAMBDA = 2.0  # backoff strength toward the class unigram distribution


def sentences(text, drop_stop):
    """Lowercased word lists per sentence; closed-class words removed if drop_stop."""
    out = []
    for s in SENT_SPLIT.split(text.lower()):
        ws = WORD.findall(s)
        if drop_stop:
            ws = [w for w in ws if w not in ENGLISH_STOP_WORDS]
        if len(ws) > 1:
            out.append(ws)
    return out


def bigrams(sents):
    """Adjacent pairs within a sentence (after stopword removal)."""
    return [(s[i], s[i + 1]) for s in sents for i in range(len(s) - 1)]


class BigramLM:
    """Counts for one class; direction 'fwd' models P(b|a), 'bwd' models P(a|b)."""

    def __init__(self):
        self.pair = Counter()    # (a, b)
        self.left = Counter()    # count of a as the left word
        self.right = Counter()   # count of b as the right word
        self.n = 0

    def add(self, bg):
        for a, b in bg:
            self.pair[(a, b)] += 1
            self.left[a] += 1
            self.right[b] += 1
        self.n += len(bg)

    def logprob(self, bg, vocab_size, direction):
        lp = 0.0
        for a, b in bg:
            if direction == "fwd":
                ctx, tgt, uni, ctx_n = a, b, self.right, self.left
                c = self.pair.get((a, b), 0)
            else:
                ctx, tgt, uni, ctx_n = b, a, self.left, self.right
                c = self.pair.get((a, b), 0)
            p_uni = (uni.get(tgt, 0) + 1) / (self.n + vocab_size)
            p = (c + LAMBDA * p_uni) / (ctx_n.get(ctx, 0) + LAMBDA)
            lp += np.log(p)
        return lp


def fit(bgs, labels, classes):
    lms = {c: BigramLM() for c in classes}
    for bg, y in zip(bgs, labels):
        lms[y].add(bg)
    return lms


def score(lms, bgs, classes, vocab_size):
    """(n_texts, n_classes, 2) array: [..., 0]=forward logprob, [..., 1]=backward."""
    S = np.zeros((len(bgs), len(classes), 2))
    for i, bg in enumerate(bgs):
        for j, c in enumerate(classes):
            S[i, j, 0] = lms[c].logprob(bg, vocab_size, "fwd")
            S[i, j, 1] = lms[c].logprob(bg, vocab_size, "bwd")
    return S


def oof_scores(bgs, y, groups, classes, n_splits=5):
    vocab_size = len({w for bg in bgs for pair in bg for w in pair})
    S = np.zeros((len(bgs), len(classes), 2))
    for tr, va in GroupKFold(n_splits).split(bgs, y, groups):
        lms = fit([bgs[i] for i in tr], y[tr], classes)
        S[va] = score(lms, [bgs[i] for i in va], classes, vocab_size)
    return S


def main(drop_stop=True):
    df = pd.read_csv(f"{DATA}/train.csv")
    classes = sorted(df.model.unique())
    cidx = {c: i for i, c in enumerate(classes)}
    y = df.model.values
    yi = np.array([cidx[c] for c in y])
    groups = df.prompt_id.values
    bgs = [bigrams(sentences(t, drop_stop)) for t in df.text]
    print(f"drop_stopwords={drop_stop} | mean bigrams/text={np.mean([len(b) for b in bgs]):.0f}")

    S = oof_scores(bgs, y, groups, classes)

    print("\nStage 1: argmax(fwd + w*bwd)   (chance = %.3f)" % (1 / len(classes)))
    for w in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        pred = (S[:, :, 0] + w * S[:, :, 1]).argmax(1)
        print(f"  w_bwd={w:<4} acc={np.mean(pred == yi):.4f}")
    print(f"  bwd only   acc={np.mean(S[:, :, 1].argmax(1) == yi):.4f}")

    # Stage 2: multinomial logistic regression on per-class scores. Scores are
    # centred per text (subtract row mean) so only the between-model differences remain.
    def feats(S, w=None):
        F = S - S.mean(1, keepdims=True)
        return F[:, :, 0] if w is None else np.hstack([F[:, :, 0], F[:, :, 1]])

    print("\nStage 2: logistic regression on OOF scores (grouped CV)")
    for name, X in [("fwd only", feats(S)), ("fwd + bwd", feats(S, w=1))]:
        accs = []
        for tr, va in GroupKFold(5).split(X, yi, groups):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), yi[tr])
            accs.append(clf.score(sc.transform(X[va]), yi[va]))
        print(f"  {name:<10} acc={np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    # Per-class accuracy of the best stage-1 setting, to see which models are confusable.
    best_w = max([0, .1, .25, .5, .75, 1.0],
                 key=lambda w: np.mean((S[:, :, 0] + w * S[:, :, 1]).argmax(1) == yi))
    pred = (S[:, :, 0] + best_w * S[:, :, 1]).argmax(1)
    print(f"\nPer-model recall at w_bwd={best_w}:")
    for c in classes:
        m = yi == cidx[c]
        print(f"  {c:<24}{np.mean(pred[m] == yi[m]):.3f}")


if __name__ == "__main__":
    main(drop_stop=("--keep-stop" not in sys.argv))
