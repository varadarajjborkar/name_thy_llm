"""TF-IDF (word + char n-grams) baselines with grouped-by-prompt CV."""
import re
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

DATA = "/Users/varadarajborkar/Desktop/call_llm/data"


def style_features(texts):
    """Cheap formatting/length statistics, which TF-IDF doesn't capture directly."""
    rows = []
    for t in texts:
        words = t.split()
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        n = max(len(t), 1)
        rows.append([
            np.log(len(t)), np.log(max(len(words), 1)),
            np.mean([len(w) for w in words]) if words else 0,
            len(words) / max(len(sents), 1),
            t.count(",") / n, t.count("—") / n, t.count("–") / n, t.count("-") / n,
            t.count(";") / n, t.count(":") / n, t.count("(") / n, t.count('"') / n,
            t.count("“") / n, t.count("’") / n, t.count("'") / n, t.count("\n") / n,
            t.count("*") / n, sum(c.isupper() for c in t) / n,
            sum(c.isdigit() for c in t) / n, sum(ord(c) > 127 for c in t) / n,
        ])
    return np.array(rows)


def make_vecs(kind):
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, lowercase=True)
    char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True,
                           min_df=2, lowercase=False, max_features=300_000)
    return {"word": [word], "char": [char], "word+char": [word, char]}[kind]


def run(kind, clf_name, texts, yi, groups, style=None, C=1.0):
    accs = []
    for tr, va in GroupKFold(5).split(texts, yi, groups):
        mats_tr, mats_va = [], []
        for v in make_vecs(kind):
            v = type(v)(**v.get_params())  # fresh copy per fold
            mats_tr.append(v.fit_transform(texts[tr]))
            mats_va.append(v.transform(texts[va]))
        if style is not None:
            sc = StandardScaler().fit(style[tr])
            mats_tr.append(sp.csr_matrix(sc.transform(style[tr]) * 0.3))
            mats_va.append(sp.csr_matrix(sc.transform(style[va]) * 0.3))
        Xtr, Xva = sp.hstack(mats_tr).tocsr(), sp.hstack(mats_va).tocsr()
        clf = (LogisticRegression(C=C, max_iter=3000) if clf_name == "logreg"
               else LinearSVC(C=C))
        accs.append(clf.fit(Xtr, yi[tr]).score(Xva, yi[va]))
    return np.mean(accs), np.std(accs)


def main():
    df = pd.read_csv(f"{DATA}/train.csv")
    classes = sorted(df.model.unique())
    yi = df.model.map({c: i for i, c in enumerate(classes)}).values
    texts, groups = df.text.values, df.prompt_id.values
    style = style_features(texts)
    print(f"chance = {1 / len(classes):.3f}\n")

    # style features alone
    accs = []
    for tr, va in GroupKFold(5).split(texts, yi, groups):
        sc = StandardScaler().fit(style[tr])
        clf = LogisticRegression(max_iter=3000).fit(sc.transform(style[tr]), yi[tr])
        accs.append(clf.score(sc.transform(style[va]), yi[va]))
    print(f"{'style only':<28}{'logreg':<8} acc={np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    configs = [("word", "logreg", None, 10), ("char", "logreg", None, 10),
               ("word+char", "logreg", None, 10), ("word+char", "svm", None, 0.5),
               ("word+char", "logreg", style, 10), ("word+char", "svm", style, 0.5)]
    for kind, clf, st, C in configs:
        t0 = time.time()
        m, s = run(kind, clf, texts, yi, groups, st, C)
        tag = kind + ("+style" if st is not None else "")
        print(f"{tag:<28}{clf:<8} C={C:<5} acc={m:.4f} +/- {s:.4f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
