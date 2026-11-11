"""Feature-tree pipeline: train.csv only, no neural network, no word lists, no TF-IDF features.

Idea (the user's design, 2026-11-11):
  1. Style features per text: punctuation, sentence rhythm, word length, structure patterns (features/style_features.py
     minus its word-list columns w_* and sentence-starter columns start_*: "favourite words" are excluded on purpose).
  2. Prompt centering of the features: texts that answer the same prompt are grouped (topic clusters, no labels needed)
     and every feature is standardised inside its group, (x - group mean) / group std. What the topic pushes up for
     everybody cancels; what this writer does differently from the other writers of the same prompt remains.
  3. Company first: a tree model picks the company (claude, gemini, gpt, glm, kimi, qwen), with an optional
     inverse-frequency weight (weight = (1 / company share) ** alpha) so the small companies (kimi, qwen: one model
     each) are not drowned by the big ones.
  4. Then the exact model inside the company, one tree per company, and a dedicated Claude+GLM tree (7 models) that
     can replace the company split between Claude and GLM, the pair we confuse most.
  5. Soft combination: P(model) = P(company) * P(model | company), so a doubtful company call keeps both options.
  6. The same post-processing as the shipped Qwen notebook: prompt centering of log-probabilities (w 0.5) and
     partial Sinkhorn (0.45), only on a test set of at least 180 rows.

TF-IDF appears in exactly one place: find_groups, to find which texts share a prompt (topic). It never becomes a
feature. Self-contained apart from style_features (the Kaggle kernel builder pastes that file above this one).
"""
import numpy as np
import pandas as pd

from style_features import NAMES, extract  # BUILD: drop (the kernel builder pastes style_features.py above)

FAMILIES = ["claude", "gemini", "gpt", "glm", "kimi", "qwen"]
KEEP = [i for i, n in enumerate(NAMES) if not n.startswith(("w_", "start_"))]
FEAT_NAMES = [NAMES[i] for i in KEEP]
CENTER_W = 0.5      # post-processing, as in kernel_qwen/ntl_qwen_blend.py
SINK_ALPHA = 0.45
MIN_POST_ROWS = 180  # 10 x 18 classes, as in the notebook


def fam_of(model):
    return next(f for f in FAMILIES if model.startswith(f))


def featurize(texts):
    """Style feature matrix (n, len(FEAT_NAMES)). No missing values: extract() always returns numbers
    (curly_quote_share is -1 when a text has no quotes at all; trees treat that as its own value)."""
    F = np.array([extract(t) for t in texts], dtype=np.float64)[:, KEEP]
    assert np.isfinite(F).all(), "non-finite style feature"
    return F


def find_groups(texts, n_per_group=18, seed=0):
    """Prompt groups without labels: TF-IDF on topic words (very common and stop words removed), SVD, cosine
    average-linkage clustering into len/18 clusters. Same method as the shipped notebook (97% purity on train.csv).
    Rows with no usable topic word get their own singleton group."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    n = len(texts)
    groups = np.arange(n) + n  # singletons that never collide with cluster ids
    if n < 2 * n_per_group:
        return groups
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, max_df=0.05).fit_transform(texts)
    ok = np.asarray(X.getnnz(axis=1) > 0)
    if ok.sum() >= 2 * n_per_group:
        Z = normalize(TruncatedSVD(min(256, X.shape[1] - 1, int(ok.sum()) - 1), random_state=seed).fit_transform(X[ok]))
        groups[ok] = AgglomerativeClustering(n_clusters=max(1, round(ok.sum() / n_per_group)), metric="cosine",
                                             linkage="average").fit_predict(Z)
    return groups


def center(F, groups):
    """Within-group standardisation, (x - group mean) / (group std + eps). Groups of one text give 0.
    eps = 5% of the feature's overall std, so a feature that is nearly constant inside a group can't explode."""
    out = np.zeros_like(F)
    eps = 0.05 * F.std(0) + 1e-9
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if len(idx) < 2:
            continue
        sub = F[idx]
        out[idx] = (sub - sub.mean(0)) / (sub.std(0) + eps)
    return out


def build_X(F, groups, mode):
    """Rounded to 9 decimals: float dust of ~1e-14 (same texts, different summation order) changed LightGBM's bins
    for tied values and moved a fold by about 1 point; rounding makes every code path give bit-identical inputs."""
    if mode == "raw":
        X = F
    elif mode == "centered":
        X = center(F, groups)
    elif mode == "both":
        X = np.hstack([F, center(F, groups)])
    else:
        raise ValueError(mode)
    return np.round(X, 9)


# ---------------------------------------------------------------- tree models
LGBM_SEEDS = 3  # one LightGBM fit moved by about +-1 point from noise alone; the average of 3 seeds is steadier


class SeedAverage:
    """Average of the same LightGBM settings trained with several seeds (bagging and column sampling differ)."""

    def __init__(self, makers):
        self.models = [mk() for mk in makers]

    def fit(self, X, y, sample_weight=None):
        for m in self.models:
            m.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.models[0].classes_
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


def make_model(kind, p, seed=0, n_jobs=6):
    if kind == "lgbm":
        import lightgbm as lgb

        def one(s):
            return lambda: lgb.LGBMClassifier(
                n_estimators=p["n"], learning_rate=p["lr"], num_leaves=p["leaves"], min_child_samples=p["min_child"],
                colsample_bytree=p["ff"], subsample=0.8, subsample_freq=1, reg_lambda=p["l2"], verbose=-1,
                n_jobs=n_jobs, random_state=s)
        return SeedAverage([one(seed + i) for i in range(LGBM_SEEDS)])
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    cls = RandomForestClassifier if kind == "rf" else ExtraTreesClassifier
    return cls(n_estimators=p.get("n", 500), max_features=p["mf"], min_samples_leaf=p["leaf"], max_depth=p["depth"],
               n_jobs=n_jobs, random_state=seed)


def node_classes(node, classes18):
    if node == "top":
        return FAMILIES
    if node == "cg":
        return [c for c in classes18 if fam_of(c) in ("claude", "glm")]
    return [c for c in classes18 if fam_of(c) == node]


def node_target(node, models):
    """Label of each training text for this node (company for 'top', the model itself otherwise)."""
    return np.array([fam_of(m) for m in models]) if node == "top" else np.asarray(models)


def node_rows(node, models):
    if node == "top":
        return np.ones(len(models), dtype=bool)
    fams = np.array([fam_of(m) for m in models])
    return np.isin(fams, ["claude", "glm"]) if node == "cg" else fams == node


def fit_node(node, cfg, X, models, classes18, seed=0, n_jobs=6):
    """Fit one node on its rows. cfg = {"kind", "params", "alpha"}; alpha only matters for 'top'.
    Returns (model, class list in predict_proba column order)."""
    rows = node_rows(node, models)
    y_txt = node_target(node, np.asarray(models)[rows])
    cl = node_classes(node, classes18)
    y = np.array([cl.index(v) for v in y_txt])
    w = np.ones(len(y))
    if node == "top" and cfg.get("alpha", 0):
        share = np.bincount(y, minlength=len(cl)) / len(y)
        w = (1.0 / share[y]) ** cfg["alpha"]
        w *= len(w) / w.sum()
    m = make_model(cfg["kind"], cfg["params"], seed=seed, n_jobs=n_jobs)
    m.fit(X[rows], y, sample_weight=w)
    assert list(m.classes_) == list(range(len(cl))), f"node {node}: a class is missing from its training rows"
    return m, cl


def predict_node(fitted, X):
    m, _ = fitted
    p = np.clip(m.predict_proba(X), 1e-6, 1.0)
    return p / p.sum(1, keepdims=True)


def combine(P_top, P_sub, classes18, use_cg):
    """P(model) = P(company) * P(model | company). With use_cg the Claude+GLM tree splits the joint Claude+GLM mass.
    P_sub: dict node -> probability matrix in node_classes(node) column order."""
    P = np.zeros((P_top.shape[0], len(classes18)))
    for fi, f in enumerate(FAMILIES):
        cols = [classes18.index(c) for c in node_classes(f, classes18)]
        if use_cg and f in ("claude", "glm"):
            continue
        if len(cols) == 1:
            P[:, cols[0]] = P_top[:, fi]
        else:
            P[:, cols] = P_top[:, [fi]] * P_sub[f]
    if use_cg:
        cols = [classes18.index(c) for c in node_classes("cg", classes18)]
        mass = P_top[:, FAMILIES.index("claude")] + P_top[:, FAMILIES.index("glm")]
        P[:, cols] = mass[:, None] * P_sub["cg"]
    return P


# ---------------------------------------------------------------- post-processing (same as the shipped notebook)
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


def post(logp, groups):
    """Prompt centering of log-probabilities (w CENTER_W, groups of 2+), then partial Sinkhorn (SINK_ALPHA).
    Only on at least MIN_POST_ROWS rows, as in the notebook."""
    if len(logp) < MIN_POST_ROWS:
        return logp
    lb = log_softmax(logp)
    out = lb.copy()
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= 2:
            out[m] = lb[m] - CENTER_W * lb[m].mean(0)
    out = log_softmax(out)
    return (1 - SINK_ALPHA) * out + SINK_ALPHA * sinkhorn(out)
