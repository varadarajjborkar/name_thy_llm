"""Group-wise assignment: passages that answer the same prompt get one model each.

find_groups(texts)        recover prompt groups from content words (no labels, no prompt_id needed)
assign_groups(lp, groups) per group, Hungarian assignment maximising total log-probability, each model used at most
                          ceil(n/18) times in a group of n passages. Self-contained so the Kaggle notebook can copy it.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


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


def find_groups(texts, n_per_group=18, dims=256, seed=0):
    """Topic clusters: TF-IDF on content words (style words removed), SVD, cosine average-linkage clustering."""
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, max_df=0.05).fit_transform(texts)
    k = min(dims, X.shape[1] - 1, X.shape[0] - 1)
    Z = normalize(TruncatedSVD(k, random_state=seed).fit_transform(X))
    n_groups = max(1, round(len(texts) / n_per_group))
    return AgglomerativeClustering(n_clusters=n_groups, metric="cosine", linkage="average").fit_predict(Z)


def assign_groups(lp, groups):
    lp = log_softmax(lp)
    n_cls = lp.shape[1]
    pred = lp.argmax(1).copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        reps = int(np.ceil(len(idx) / n_cls))
        rows, cols = linear_sum_assignment(-np.tile(lp[idx], (1, reps)))
        pred[idx[rows]] = cols % n_cls
    return pred
