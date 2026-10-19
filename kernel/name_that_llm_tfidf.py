"""Name That LLM: TF-IDF (word + char) + style features, LinearSVC. Self-contained.

Written for a hidden rerun: test.csv has ~4000 rows there (3 in the interactive run), and
sample_submission.parquet is not relied on, so a mismatching sample cannot break the run.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

STYLE_W, C = 0.3, 1.0  # best config from grouped-by-prompt CV (~36.8% accuracy)
ROOT = os.environ.get("NTL_ROOT", "/kaggle/input")
LOCAL_FALLBACK = "/Users/varadarajborkar/Desktop/call_llm/data"


def data_dir():
    """Directory holding train.csv: the known competition path first, then a search."""
    for d in (f"{ROOT}/competitions/name-that-llm", f"{ROOT}/name-that-llm"):
        if os.path.exists(f"{d}/train.csv"):
            return d
    hits = sorted(glob.glob(f"{ROOT}/**/train.csv", recursive=True))
    return os.path.dirname(hits[0]) if hits else LOCAL_FALLBACK


def clean(texts):
    """Missing/non-string passages become empty strings instead of crashing the vectorizers."""
    return pd.Series(texts).fillna("").astype(str).values


def style_features(texts):
    rows = []
    for t in texts:
        words = t.split()
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        n = max(len(t), 1)
        rows.append([
            np.log(max(len(t), 1)), np.log(max(len(words), 1)),
            np.mean([len(w) for w in words]) if words else 0,
            len(words) / max(len(sents), 1),
            t.count(",") / n, t.count("—") / n, t.count("–") / n, t.count("-") / n,
            t.count(";") / n, t.count(":") / n, t.count("(") / n, t.count('"') / n,
            t.count("“") / n, t.count("’") / n, t.count("'") / n, t.count("\n") / n,
            t.count("*") / n, sum(c.isupper() for c in t) / n,
            sum(c.isdigit() for c in t) / n, sum(ord(c) > 127 for c in t) / n,
        ])
    return np.array(rows)


d = data_dir()
train = pd.read_csv(f"{d}/train.csv")
test_path = f"{d}/test.csv" if os.path.exists(f"{d}/test.csv") else f"{LOCAL_FALLBACK}/test.csv"
test = pd.read_csv(test_path)
train_text, test_text = clean(train.text), clean(test.text)
print("data dir:", d, "| train", train.shape, "| test", test.shape,
      "| empty test texts:", int((pd.Series(test_text).str.strip() == "").sum()), flush=True)

word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True,
                       min_df=2, lowercase=False, max_features=300_000)
scaler = StandardScaler().fit(style_features(train_text))


def featurize(texts, fit=False):
    w = word.fit_transform(texts) if fit else word.transform(texts)
    c = char.fit_transform(texts) if fit else char.transform(texts)
    s = sp.csr_matrix(scaler.transform(style_features(texts)) * STYLE_W)
    return sp.hstack([w, c, s]).tocsr()


clf = LinearSVC(C=C, max_iter=2000).fit(featurize(train_text, fit=True), train.model.values)
pred = clf.predict(featurize(test_text))

sub = pd.DataFrame({"id": test.id.astype(str).values, "model": pred})
assert len(sub) == len(test), "row count changed"
assert sub.model.isin(train.model.unique()).all(), "unknown label predicted"

out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("NTL_OUT", ".")
sub.to_parquet(f"{out_dir}/submission.parquet", index=False)
print(sub.model.value_counts().to_string(), "\nwritten", len(sub), "rows to", f"{out_dir}/submission.parquet")
sys.stdout.flush()
