"""Name That LLM: TF-IDF SVM + LightGBM style-feature blend. CPU only, internet off.

Members:
  * TF-IDF (word 1-2 + char 2-5) + 20 style stats, LinearSVC, fit here on the host train.csv.
  * LightGBM on 316 hand-made style features, trained offline on 100k external passages (dataset ntl-lgbm-style).
Blend: 0.5 * log_softmax(3 * svm_scores) + 0.5 * log(lgbm_probs). Weights fixed in advance;
grouped-by-prompt CV on train.csv: SVM 0.368, LightGBM 0.365, blend 0.419.

Written for the hidden rerun (~4000 test rows instead of 3): ids come from test.csv, missing text is tolerated,
and nothing depends on sample_submission.parquet.
"""
import glob
import os
import re
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = os.environ.get("NTL_ROOT", "/kaggle/input")
W_SVM, T_SVM = 0.5, 3.0


def find(pattern):
    hits = sorted(glob.glob(f"{ROOT}/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under {ROOT}")
    return hits[0]


def clean(texts):
    return pd.Series(texts).fillna("").astype(str).values


def style_stats(texts):  # the 20 stats used by the original TF-IDF notebook
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


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


# ---------------------------------------------------------------- data
train_csv = find("train.csv")
d = os.path.dirname(train_csv)
train = pd.read_csv(train_csv)
test = pd.read_csv(f"{d}/test.csv")
train_text, test_text = clean(train.text), clean(test.text)
classes = sorted(train.model.unique())
print("data:", d, "| train", train.shape, "| test", test.shape, flush=True)

# ---------------------------------------------------------------- member 1: TF-IDF SVM
word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True, min_df=2, lowercase=False,
                       max_features=300_000)
scaler = StandardScaler().fit(style_stats(train_text))


def featurize(texts, fit=False):
    w = word.fit_transform(texts) if fit else word.transform(texts)
    c = char.fit_transform(texts) if fit else char.transform(texts)
    return sp.hstack([w, c, sp.csr_matrix(scaler.transform(style_stats(texts)) * 0.3)]).tocsr()


svm = LinearSVC(C=1.0, max_iter=2000).fit(featurize(train_text, fit=True), train.model.values)
assert list(svm.classes_) == classes, "SVM class order differs"
svm_lp = log_softmax(T_SVM * svm.decision_function(featurize(test_text)))
print("svm done", flush=True)

# ---------------------------------------------------------------- member 2: LightGBM on style features
model_file = find("lgbm_ext.txt")
sys.path.insert(0, os.path.dirname(find("style_features.py")))
from style_features import NAMES, extract  # noqa: E402

booster = lgb.Booster(model_file=model_file)
assert booster.num_feature() == len(NAMES), (booster.num_feature(), len(NAMES))
Xs = pd.DataFrame(np.asarray([extract(t) for t in test_text], dtype=np.float32), columns=NAMES)
lgbm_p = booster.predict(Xs, num_iteration=booster.best_iteration or None)
assert lgbm_p.shape == (len(test), len(classes)), lgbm_p.shape
print("lightgbm done", flush=True)

# ---------------------------------------------------------------- blend + write
blend = W_SVM * svm_lp + (1 - W_SVM) * np.log(np.clip(lgbm_p, 1e-9, 1.0))
pred = np.array(classes)[blend.argmax(1)]
sub = pd.DataFrame({"id": test.id.astype(str).values, "model": pred})
assert len(sub) == len(test) and sub.id.notna().all(), "row count changed"
out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("NTL_OUT", ".")
sub.to_parquet(f"{out_dir}/submission.parquet", index=False)
print(sub.model.value_counts().to_string(), "\nwritten", len(sub), "rows", flush=True)
