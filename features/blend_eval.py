"""Honest estimate of a TF-IDF SVM + LightGBM style blend on the host train.csv.

TF-IDF SVM: out-of-fold decision scores, 5 folds grouped by prompt (refit per fold, same features as the notebook).
LightGBM: trained on external data only (features/lgbm_run.py ext), so its train.csv predictions are already unseen.
Blend = w * log_softmax(T * svm_scores) + (1 - w) * log(lgbm_probs), optionally class-balanced (Sinkhorn).
A default (w=0.5, T=3) is declared up front; the best grid point is shown only to see how much tuning flatters it.
"""
import ctypes
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
try:  # stay polite to the GPU training run
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception:
    pass


def style_features(texts):  # identical to kernel/name_that_llm_tfidf.py
    rows = []
    for t in texts:
        words = t.split()
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        n = max(len(t), 1)
        rows.append([np.log(max(len(t), 1)), np.log(max(len(words), 1)),
                     np.mean([len(w) for w in words]) if words else 0, len(words) / max(len(sents), 1),
                     t.count(",") / n, t.count("—") / n, t.count("–") / n, t.count("-") / n,
                     t.count(";") / n, t.count(":") / n, t.count("(") / n, t.count('"') / n,
                     t.count("“") / n, t.count("’") / n, t.count("'") / n, t.count("\n") / n,
                     t.count("*") / n, sum(c.isupper() for c in t) / n,
                     sum(c.isdigit() for c in t) / n, sum(ord(c) > 127 for c in t) / n])
    return np.array(rows)


def svm_scores(tr_text, tr_y, te_text):
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True, min_df=2, lowercase=False,
                           max_features=300_000)
    sc = StandardScaler().fit(style_features(tr_text))
    f = lambda t, fit: sp.hstack([word.fit_transform(t) if fit else word.transform(t),
                                  char.fit_transform(t) if fit else char.transform(t),
                                  sp.csr_matrix(sc.transform(style_features(t)) * 0.3)]).tocsr()
    clf = LinearSVC(C=1.0, max_iter=2000).fit(f(tr_text, True), tr_y)
    return clf.decision_function(f(te_text, False)), list(clf.classes_)


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def sinkhorn(lp, iters=100):
    p = np.exp(lp - lp.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    for _ in range(iters):
        p /= p.sum(0, keepdims=True)
        p /= p.sum(1, keepdims=True)
    return np.log(p + 1e-12)


df = pd.read_csv(ROOT / "data/train.csv")
text = df.text.fillna("").astype(str).values
classes = sorted(df.model.unique())
y = df.model.map({c: i for i, c in enumerate(classes)}).to_numpy()
oof = np.zeros((len(df), len(classes)))
for k, (tr, va) in enumerate(GroupKFold(5).split(df, y, df.prompt_id)):
    s, cl = svm_scores(text[tr], df.model.values[tr], text[va])
    assert cl == classes
    oof[va] = s
    print(f"fold {k} svm {bacc(y[va], s.argmax(1)):.4f}", flush=True)
lg = np.load(ROOT / "data/processed/lgbm_ext_val_probs.npy")
lgl = np.log(np.clip(lg, 1e-9, 1))

print(f"\nSVM alone (OOF)        {bacc(y, oof.argmax(1)):.4f}   balanced {bacc(y, sinkhorn(log_softmax(3 * oof)).argmax(1)):.4f}")
print(f"LightGBM alone (unseen) {bacc(y, lg.argmax(1)):.4f}   balanced {bacc(y, sinkhorn(lgl).argmax(1)):.4f}")
res = {}
for w in (0.3, 0.5, 0.7):
    for T in (1, 3, 10):
        b = w * log_softmax(T * oof) + (1 - w) * lgl
        res[(w, T)] = (bacc(y, b.argmax(1)), bacc(y, sinkhorn(b).argmax(1)))
d = res[(0.5, 3)]
best = max(res, key=lambda k: res[k][1])
print(f"\nDEFAULT blend w=0.5 T=3 {d[0]:.4f}   balanced {d[1]:.4f}")
print(f"best grid point w={best[0]} T={best[1]}  {res[best][0]:.4f}   balanced {res[best][1]:.4f}  (optimistic: chosen on these rows)")
