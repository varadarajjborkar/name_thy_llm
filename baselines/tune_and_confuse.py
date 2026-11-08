"""Tune C, test adding bigram-LM scores, and inspect confusions. Grouped 5-fold CV."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from bigram_lm import bigrams, oof_scores, sentences
from tfidf_baseline import DATA, style_features

df = pd.read_csv(f"{DATA}/train.csv")
classes = sorted(df.model.unique())
yi = df.model.map({c: i for i, c in enumerate(classes)}).values
texts, groups = df.text.values, df.prompt_id.values
style = style_features(texts)

# bigram-LM OOF scores (stopwords kept, which scored better), centred per text
bgs = [bigrams(sentences(t, drop_stop=False)) for t in texts]
S = oof_scores(bgs, df.model.values, groups, classes)
S = S - S.mean(1, keepdims=True)
lm_feats = np.hstack([S[:, :, 0], S[:, :, 1]])

# cache fold features once
folds = []
for tr, va in GroupKFold(5).split(texts, yi, groups):
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True,
                           min_df=2, lowercase=False, max_features=300_000)
    ss, ls = StandardScaler().fit(style[tr]), StandardScaler().fit(lm_feats[tr])
    parts = lambda idx, fit: dict(
        tfidf=sp.hstack([(word.fit_transform if fit else word.transform)(texts[idx]),
                         (char.fit_transform if fit else char.transform)(texts[idx])]).tocsr(),
        style=sp.csr_matrix(ss.transform(style[idx])),
        lm=sp.csr_matrix(ls.transform(lm_feats[idx])))
    folds.append((tr, va, parts(tr, True), parts(va, False)))
print("features cached", flush=True)


def evaluate(clf_fn, style_w, lm_w, return_pred=False):
    accs, pred = [], np.zeros(len(yi), int)
    for tr, va, ptr, pva in folds:
        build = lambda p: sp.hstack([p["tfidf"], p["style"] * style_w, p["lm"] * lm_w]).tocsr()
        clf = clf_fn().fit(build(ptr), yi[tr])
        pred[va] = clf.predict(build(pva))
        accs.append(np.mean(pred[va] == yi[va]))
    return (np.mean(accs), np.std(accs), pred) if return_pred else (np.mean(accs), np.std(accs))


# Earlier sweep (style_w 0.3/0.6/1.0 x C 0.1/0.3/1.0): best was style_w=0.3, C=1.0 (36.8%);
# accuracy rose with C and fell with style_w; style_w=1.0 stopped converging.
print("\nSVM: larger C (style weight 0.3, lm weight 0)")
for C in [1.0, 3.0]:
    m, s = evaluate(lambda: LinearSVC(C=C, max_iter=2000), 0.3, 0)
    print(f"  C={C:<4} acc={m:.4f} +/- {s:.4f}", flush=True)

print("\nAdding bigram-LM scores (SVM, C=1, style weight 0.3)")
for lw in [0.1, 0.3]:
    m, s = evaluate(lambda: LinearSVC(C=1.0, max_iter=2000), 0.3, lw)
    print(f"  lm_w={lw:<4} acc={m:.4f} +/- {s:.4f}", flush=True)

# confusion for the best config so far
m, s, pred = evaluate(lambda: LinearSVC(C=1.0, max_iter=2000), 0.3, 0, return_pred=True)
cm = pd.crosstab(pd.Series(yi, name="true"), pd.Series(pred, name="pred")).reindex(
    index=range(len(classes)), columns=range(len(classes)), fill_value=0)
print(f"\nConfusion for SVM C=1 style_w=0.3 (acc={m:.4f})")
print(f"{'model':<24}{'recall':>7}  top confusions")
for i, c in enumerate(classes):
    row = cm.loc[i].values.copy()
    rec = row[i] / row.sum()
    row[i] = 0
    top = np.argsort(row)[::-1][:2]
    print(f"{c:<24}{rec:>7.2f}  " + ", ".join(f"{classes[j]} ({row[j] / cm.loc[i].sum():.0%})" for j in top))
fam = lambda c: c.split("-")[0]
fam_true = np.array([fam(classes[i]) for i in yi])
fam_pred = np.array([fam(classes[i]) for i in pred])
print(f"\nFamily-level accuracy (claude/gemini/gpt/glm/kimi/qwen): {np.mean(fam_true == fam_pred):.4f}")
