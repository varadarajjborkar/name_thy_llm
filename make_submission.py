"""Fit the best TF-IDF + style SVM on all of train and write a submission parquet."""
import os

import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from tfidf_baseline import DATA, style_features

OUT = "/Users/varadarajborkar/Desktop/call_llm/submissions"
STYLE_W, C = 0.3, 1.0  # best config from grouped CV (~36.8%)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
sample = pd.read_parquet(f"{DATA}/sample_submission.parquet")

word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True,
                       min_df=2, lowercase=False, max_features=300_000)
scaler = StandardScaler().fit(style_features(train.text.values))


def featurize(texts, fit=False):
    w = word.fit_transform(texts) if fit else word.transform(texts)
    c = char.fit_transform(texts) if fit else char.transform(texts)
    s = sp.csr_matrix(scaler.transform(style_features(texts)) * STYLE_W)
    return sp.hstack([w, c, s]).tocsr()


clf = LinearSVC(C=C, max_iter=2000).fit(featurize(train.text.values, fit=True), train.model.values)
pred = clf.predict(featurize(test.text.values))

sub = pd.DataFrame({"id": test.id.values, "model": pred})

# validate against the sample submission
assert list(sub.columns) == list(sample.columns), "column mismatch"
assert set(sub.id) == set(sample.id) and len(sub) == len(sample), "id mismatch"
assert sub.id.is_unique and sub.model.isin(train.model.unique()).all(), "bad ids/labels"
sub = sub.set_index("id").loc[sample.id].reset_index()  # same row order as the sample

os.makedirs(OUT, exist_ok=True)
sub.to_parquet(f"{OUT}/submission_tfidf_svm.parquet", index=False)
print(sub.to_string())
print("\nwritten:", f"{OUT}/submission_tfidf_svm.parquet", "| dtypes:", dict(sub.dtypes.astype(str)),
      "| sample dtypes:", dict(sample.dtypes.astype(str)))
