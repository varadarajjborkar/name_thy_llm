"""Which kind of signal can tell the COMPANY (claude, gemini, gpt, glm, kimi, qwen) from train.csv alone?
Measurement only (nothing here ships). Company balanced accuracy, folds 0-3 out of fold, GroupKFold(5) by prompt_id,
class_weight balanced (= the inverse-frequency weighting) everywhere.

Feature sets
  style_c    the tree pipeline's prompt-centered style features + LightGBM (reference, about 0.57)
  words      word 1-2 gram TF-IDF + logistic regression (Sun et al. 2025 baseline: 85.5% with 10K texts per class)
  words_c    the same words, prompt-centered: each text's vector minus its prompt group's mean (topic words cancel)
  funcw      only English function/stop words (no topic words at all), counts per word
  funcw_c    function words, prompt-centered
  chars      character 2-5 grams inside words
  first3     only the first 3 words of the text (the paper: the first word alone carries signal)
  combos     averages of log-probabilities of the above
Run: .venv/Scripts/python.exe research/company_probe.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "features"), str(ROOT / "tree")]
import ntl_tree as T  # noqa: E402

try:
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

train = pd.read_csv(ROOT / "data/train.csv")
texts = train.text.fillna("").astype(str).to_numpy()
yfam = np.array([T.FAMILIES.index(T.fam_of(m)) for m in train.model])
fold = np.empty(len(train), dtype=int)
for k, (_, va) in enumerate(GroupKFold(5).split(train, train.model, train.prompt_id)):
    fold[va] = k
VAL = [0, 1, 2, 3]
SPL = {}
for k in VAL:
    tr, va = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
    SPL[k] = (tr, va, T.find_groups(texts[tr].tolist()), T.find_groups(texts[va].tolist()))
VA = np.concatenate([SPL[k][1] for k in VAL])
F = T.featurize(texts)


def gcenter(X, g):
    """Dense rows minus their group's mean row (groups of one are left as they are)."""
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)
    out = X.copy()
    for gi in np.unique(g):
        m = g == gi
        if m.sum() >= 2:
            out[m] = X[m] - X[m].mean(0)
    return out


def run_text(make_vec, centered, C):
    """OOF log-probabilities of a linear model on a text vectoriser fitted on the training side only."""
    out = []
    for k in VAL:
        tr, va, gtr, gva = SPL[k]
        vec = make_vec().fit(texts[tr])
        Xtr, Xva = vec.transform(texts[tr]), vec.transform(texts[va])
        if centered:
            Xtr, Xva = gcenter(Xtr, gtr), gcenter(Xva, gva)
        lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        lr.fit(Xtr, yfam[tr])
        out.append(lr.predict_log_proba(Xva))
    return np.vstack(out)


def run_style():
    cfg = {"kind": "lgbm", "params": dict(n=150, lr=0.05, leaves=15, min_child=20, ff=0.8, l2=5), "alpha": 1.0}
    out = []
    for k in VAL:
        tr, va, gtr, gva = SPL[k]
        fitted = T.fit_node("top", cfg, T.build_X(F[tr], gtr, "centered"), train.model.to_numpy()[tr], [], n_jobs=4)
        out.append(np.log(T.predict_node(fitted, T.build_X(F[va], gva, "centered"))))
    return np.vstack(out)


def score(lp):
    p = lp.argmax(1)
    return bacc(yfam[VA], p), [bacc(yfam[VA][fold[VA] == k], p[fold[VA] == k]) for k in VAL]


R = {}
R["style_c"] = run_style()
words = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=10000, sublinear_tf=True)
funcw = lambda: CountVectorizer(vocabulary=sorted(ENGLISH_STOP_WORDS))
chars = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=10000, sublinear_tf=True)
first3 = lambda: TfidfVectorizer(preprocessor=lambda s: " ".join(s.lower().split()[:3]), ngram_range=(1, 3), min_df=2)
for name, mk, cen in [("words", words, False), ("words_c", words, True), ("funcw", funcw, False),
                      ("funcw_c", funcw, True), ("chars", chars, False), ("chars_c", chars, True),
                      ("first3", first3, False)]:
    best = None
    for C in [1.0, 4.0, 16.0]:
        lp = run_text(mk, cen, C)
        s = score(lp)[0]
        print(f"  {name:8s} C {C:4.0f}: company bal. acc {s:.4f}", flush=True)
        if best is None or s > best[0]:
            best = (s, lp, C)
    R[name] = best[1]

print("\ncompany balanced accuracy, folds 0-3 out of fold (C picked per set, so slightly optimistic):")
for name, lp in R.items():
    s, f = score(lp)
    print(f"  {name:28s} {s:.4f}  folds " + " ".join(f"{v:.3f}" for v in f))
norm = {k: v - np.log(np.exp(v).sum(1, keepdims=True)) for k, v in R.items()}
for combo in [("words_c", "style_c"), ("words_c", "chars_c", "style_c"), ("words_c", "funcw_c", "chars_c", "style_c"),
              ("words", "chars", "style_c")]:
    s, f = score(sum(norm[c] for c in combo))
    print(f"  {' + '.join(combo):28s} {s:.4f}  folds " + " ".join(f"{v:.3f}" for v in f))
best = max(((" + ".join(c), sum(norm[x] for x in c)) for c in [("words_c", "chars_c", "style_c")]), key=lambda t: 0)
p = best[1].argmax(1)
print("\ncompany confusion for", best[0], "(rows true, columns guessed):")
print(pd.crosstab(pd.Series(np.array(T.FAMILIES)[yfam[VA]], name="true"),
                  pd.Series(np.array(T.FAMILIES)[p], name="guess")).to_string())
