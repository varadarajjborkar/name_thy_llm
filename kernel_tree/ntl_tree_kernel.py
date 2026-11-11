# AUTO-BUILT by tools/build_tree_kernel.py: edit features/style_features.py, tree/ntl_tree.py,
# tree/best_config.json or tree/kernel_main.py instead, then rebuild.
# ======== features/style_features.py
"""Hand-made stylometric features for LLM authorship (the "tabular" view of a passage).

Pure Python (re + math only) so the exact same code can be pasted into the Kaggle notebook.
extract(text) -> list of floats, aligned with NAMES. Rates are per 100 characters or per 100 words,
so passages of different length stay comparable.

Why these: LLMs differ in punctuation habits (em dashes, curly quotes, semicolons), sentence rhythm,
function-word and connective use ("however", "moreover"), list and markdown habits, and pet words.
"""
import math
import re

PUNCT = [",", ".", ";", ":", "!", "?", "—", "–", "-", "(", ")", '"', "'", "“", "”", "‘",
         "’", "…", "*", "#", "/", "%", "$", "&", "@", "[", "]", "{", "}", "_", "~", "`", "|", "=", "+",
         "<", ">", "\\", "→", "•", "·", "\n"]
PUNCT_NAME = {",": "comma", ".": "period", ";": "semicolon", ":": "colon", "!": "exclaim", "?": "question",
              "—": "emdash", "–": "endash", "-": "hyphen", "(": "lparen", ")": "rparen", '"': "dquote",
              "'": "squote", "“": "ldquo", "”": "rdquo", "‘": "lsquo", "’": "rsquo",
              "…": "ellipsis", "*": "star", "#": "hash", "/": "slash", "%": "percent", "$": "dollar",
              "&": "amp", "@": "at", "[": "lbracket", "]": "rbracket", "{": "lbrace", "}": "rbrace",
              "_": "underscore", "~": "tilde", "`": "backtick", "|": "pipe", "=": "equals", "+": "plus",
              "<": "lt", ">": "gt", "\\": "backslash", "→": "arrow", "•": "bullet", "·": "middot",
              "\n": "newline"}

# function words, connectives and hedges (rate per 100 words)
WORDS = sorted(set("""the of and a to in is that it as with for was on are by this but not or from at an be have which
you we i they their its can more than also while however moreover furthermore thus therefore yet even just still
only every each both between through across within without into over under about these those there when where who
whose what how why if so such like often rather perhaps ultimately indeed whether might may could would should must
will does do has had been being one all some many most other any no never always quietly simply truly really
essentially fundamentally arguably remarkably notably crucially additionally ultimately tapestry delve testament
landscape realm nuanced intricate crucial pivotal foster navigate underscore embrace vibrant journey beacon
symphony whisper echo fascinating profoundly seamless robust leverage human world time life people way
something nothing everything here now then first second third finally instead whereas although though because
since unless until once again already often sometimes usually less little much very quite almost nearly""".split()))

STARTERS = ["the", "it", "in", "this", "but", "and", "a", "we", "i", "they", "when", "if", "as", "for", "one", "what",
            "there", "yet", "so", "still", "perhaps", "not", "no", "that", "these", "to", "at", "on", "by", "with"]

RE_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")  # unicode letters, so "cafe" with an accent stays one word
RE_SENT = re.compile(r"(?<=[.!?])[\"'”’)]*\s+(?=[\"'“‘(\[]*[A-Z0-9])")
RE_VOWELS = re.compile(r"[aeiouy]+")
PATTERNS = {
    "not_just": re.compile(r"\bnot (?:just|only|merely|simply)\b", re.I),
    "not_but": re.compile(r"\bnot\b[^.!?]{0,50}\bbut\b", re.I),
    "its_not": re.compile(r"\b(?:it|that|this)(?:['’]s| is) not\b", re.I),
    "contraction": re.compile(r"\w['’](?:t|s|re|ve|ll|d|m)\b", re.I),
    "oxford": re.compile(r",\s+(?:and|or)\s", re.I),
    "ly_adv": re.compile(r"\b\w{3,}ly\b", re.I),
    "ing": re.compile(r"\b\w{3,}ing\b", re.I),
    "of_the": re.compile(r"\bof the\b", re.I),
    "spaced_emdash": re.compile(r" — "),
    "tight_emdash": re.compile(r"\w—\w"),
    "spaced_hyphen": re.compile(r" - "),
    "double_hyphen": re.compile(r"--"),
    "three_dots": re.compile(r"\.\.\."),
    "double_space": re.compile(r"  "),
    "space_before_punct": re.compile(r" [,.;:!?]"),
    "no_space_after_period": re.compile(r"\.[A-Z]"),
    "bold": re.compile(r"\*\*[^*]+\*\*"),
    "bullet_line": re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s"),
    "header_line": re.compile(r"(?m)^#{1,6}\s"),
    "digits": re.compile(r"\d"),
    "number": re.compile(r"\b\d[\d,.]*\b"),
    "year": re.compile(r"\b(?:1[5-9]|20)\d\d\b"),
    "quote_pair": re.compile(r"[\"“][^\"”]{1,200}[\"”]"),
    "paren_group": re.compile(r"\([^)]{1,200}\)"),
    "colon_list": re.compile(r":\s"),
    "semi_clause": re.compile(r";\s"),
    "rhetorical_q": re.compile(r"\?"),
    "we_us_our": re.compile(r"\b(?:we|us|our|ours)\b", re.I),
    "i_me_my": re.compile(r"\b(?:i|me|my|mine)\b", re.I),
    "you_your": re.compile(r"\b(?:you|your|yours)\b", re.I),
    "all_caps": re.compile(r"\b[A-Z]{2,}\b"),
    "title_case_run": re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+){1,}\b"),
}


def _stats(vals):
    n = len(vals)
    if not n:
        return 0.0, 0.0, 0.0, 0.0
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / n)
    return m, sd, float(max(vals)), float(min(vals))


def _build_names():
    names = ["chars", "words", "sentences", "log_chars", "mean_word_len", "std_word_len", "long_word_share",
             "very_long_word_share", "sent_len_mean", "sent_len_std", "sent_len_max", "sent_len_min", "sent_len_cv",
             "short_sent_share", "long_sent_share", "last_sent_len", "first_sent_len", "ttr", "hapax_share",
             "mean_syllables", "flesch", "upper_char_share", "cap_word_share", "nonascii_share", "alpha_share",
             "space_share", "starts_quote", "ends_period", "ends_question", "ends_quote", "ends_paren",
             "first_word_len", "comma_per_sent", "comma_per_word", "words_per_comma_clause"]
    names += [f"p_{PUNCT_NAME[c]}" for c in PUNCT]
    names += [f"pat_{k}" for k in PATTERNS]
    names += [f"w_{w}" for w in WORDS]
    names += [f"start_{w}" for w in STARTERS]
    names += ["curly_quote_share", "emdash_per_sent", "sent_starts_conj_share", "sent_starts_the_share"]
    return names


NAMES = _build_names()


def extract(text):
    n_chars = max(len(text), 1)
    words = RE_WORD.findall(text)
    lw = [w.lower() for w in words]
    nw = max(len(words), 1)
    sents = [s for s in RE_SENT.split(text.strip()) if s.strip()] or [text]
    ns = len(sents)
    sent_lens = [len(RE_WORD.findall(s)) for s in sents]
    wl = [len(w) for w in words]
    mwl, swl, _, _ = _stats(wl)
    sm, ssd, smax, smin = _stats(sent_lens)
    counts = {}
    for w in lw:
        counts[w] = counts.get(w, 0) + 1
    syll = [max(1, len(RE_VOWELS.findall(w))) for w in lw]
    msyl = sum(syll) / nw
    flesch = 206.835 - 1.015 * (nw / ns) - 84.6 * msyl
    upper = sum(1 for c in text if c.isupper())
    alpha = sum(1 for c in text if c.isalpha())
    nonascii = sum(1 for c in text if ord(c) > 127)
    spaces = text.count(" ")
    stripped = text.rstrip()
    last = stripped[-1:] if stripped else ""
    caps_mid = sum(1 for w in words[1:] if w[0].isupper())
    commas = text.count(",")

    f = [float(n_chars), float(len(words)), float(ns), math.log(n_chars), mwl, swl,
         sum(1 for x in wl if x >= 7) / nw, sum(1 for x in wl if x >= 10) / nw,
         sm, ssd, smax, smin, (ssd / sm if sm else 0.0),
         sum(1 for x in sent_lens if x < 8) / ns, sum(1 for x in sent_lens if x > 30) / ns,
         float(sent_lens[-1]), float(sent_lens[0]),
         len(counts) / nw, sum(1 for v in counts.values() if v == 1) / nw,
         msyl, flesch, upper / n_chars, caps_mid / nw, nonascii / n_chars, alpha / n_chars, spaces / n_chars,
         1.0 if text[:1] and text[:1] in "\"'“‘" else 0.0, 1.0 if last == "." else 0.0, 1.0 if last == "?" else 0.0,
         1.0 if last and last in "\"”'’" else 0.0, 1.0 if last == ")" else 0.0,
         float(len(words[0])) if words else 0.0, commas / ns, commas / nw, nw / (commas + ns)]
    f += [text.count(c) * 100.0 / n_chars for c in PUNCT]
    f += [len(p.findall(text)) * 100.0 / (n_chars if k in ("digits", "double_space", "space_before_punct",
                                                              "no_space_after_period", "three_dots") else nw)
          for k, p in PATTERNS.items()]
    f += [counts.get(w, 0) * 100.0 / nw for w in WORDS]
    starts = [(RE_WORD.findall(s) or [""])[0].lower() for s in sents]
    f += [starts.count(w) / ns for w in STARTERS]
    curly = sum(text.count(c) for c in "“”‘’")
    straight = text.count('"') + text.count("'")
    f += [curly / (curly + straight) if curly + straight else -1.0, text.count("—") / ns,
          sum(1 for w in starts if w in ("but", "and", "yet", "so", "or")) / ns, starts.count("the") / ns]
    return f

# ======== tree/ntl_tree.py
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
    try:  # 36-39 texts (max_df * n < min_df) or texts without usable words make TfidfVectorizer raise
        X = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, max_df=0.05).fit_transform(texts)
    except ValueError:
        return groups  # no grouping: features stay uncentred-equivalent (zeros) and post-processing skips groups
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

# ======== tree/best_config.json
import json
CONFIG = json.loads(r'''{
 "design": "company-first",
 "mode": "centered",
 "use_cg": false,
 "nodes": {
  "top": {
   "kind": "lgbm",
   "params": {
    "n": 150,
    "lr": 0.05,
    "leaves": 15,
    "min_child": 20,
    "ff": 0.8,
    "l2": 5
   },
   "alpha": 0.5
  },
  "claude": {
   "kind": "lgbm",
   "params": {
    "n": 150,
    "lr": 0.05,
    "leaves": 7,
    "min_child": 10,
    "ff": 0.3,
    "l2": 5
   },
   "alpha": 0
  },
  "gemini": {
   "kind": "et",
   "params": {
    "n": 500,
    "mf": 0.25,
    "leaf": 1,
    "depth": null
   },
   "alpha": 0
  },
  "gpt": {
   "kind": "lgbm",
   "params": {
    "n": 150,
    "lr": 0.05,
    "leaves": 31,
    "min_child": 10,
    "ff": 0.3,
    "l2": 5
   },
   "alpha": 0
  },
  "glm": {
   "kind": "et",
   "params": {
    "n": 500,
    "mf": "sqrt",
    "leaf": 4,
    "depth": 10
   },
   "alpha": 0
  },
  "cg": {
   "kind": "lgbm",
   "params": {
    "n": 150,
    "lr": 0.05,
    "leaves": 7,
    "min_child": 20,
    "ff": 0.8,
    "l2": 5
   },
   "alpha": 0
  }
 },
 "cv": {
  "company": 0.5713194444444445,
  "raw": 0.3814814814814815,
  "post": 0.37592592592592594,
  "folds": [
   0.3611111111111111,
   0.3592592592592593,
   0.38518518518518513,
   0.39814814814814814
  ]
 }
}''')

# ---------------------------------------------------------------- kernel main (tree/kernel_main.py)
# Runs after style_features.py, ntl_tree.py and CONFIG (pasted above by tools/build_tree_kernel.py).
# CPU only, no internet, no external data: trains the trees on train.csv and predicts test.csv.
# Local check: NTL_ROOT=<folder holding train.csv/test.csv> NTL_OUT=<out dir> NTL_PARITY_FOLD=k holds out train fold k
# (GroupKFold(5) by prompt_id) as a labelled test; the score must match research/tree_cv.py's fold k exactly.
import glob
import json
import os
import time

ROOT_IN = os.environ.get("NTL_ROOT", "/kaggle/input")
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("NTL_OUT", ".")
PARITY_FOLD = int(os.environ.get("NTL_PARITY_FOLD", "-1"))  # local pipeline check only; never in a submission
N_JOBS = int(os.environ.get("NTL_JOBS", "4"))
os.makedirs(OUT_DIR, exist_ok=True)
t_start = time.time()
SUMMARY = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "config": CONFIG, "stages": []}


def stage(name, **info):
    """Log a finished stage and rewrite run_summary.json, so even a crashed run leaves a trail."""
    SUMMARY["stages"].append({"stage": name, "t": round(time.time() - t_start, 1), **info})
    print(f"[{time.time() - t_start:6.0f}s] STAGE {name}", " ".join(f"{k}={v}" for k, v in info.items()), flush=True)
    with open(f"{OUT_DIR}/run_summary.json", "w") as f:
        json.dump(SUMMARY, f, indent=1, default=str)


hits = [h for h in sorted(glob.glob(f"{ROOT_IN}/**/train.csv", recursive=True)) if "name-that-llm" in h
        or os.environ.get("NTL_ROOT")]
assert hits, f"train.csv not found under {ROOT_IN}"
d = os.path.dirname(hits[0])
train = pd.read_csv(f"{d}/train.csv")
test = pd.read_csv(f"{d}/test.csv")
if PARITY_FOLD >= 0:
    from sklearn.model_selection import GroupKFold
    fold = np.empty(len(train), dtype=int)
    for k, (_, va) in enumerate(GroupKFold(5).split(train, train.model, train.prompt_id)):
        fold[va] = k
    test = train[fold == PARITY_FOLD].reset_index(drop=True)
    train = train[fold != PARITY_FOLD].reset_index(drop=True)
train_text = train.text.fillna("").astype(str).tolist()
test_text = test.text.fillna("").astype(str).tolist()
models = train.model.to_numpy()
classes = sorted(train.model.unique())
assert len(classes) == 18, f"expected 18 models, got {len(classes)}"
stage("data", train=len(train), test=len(test), parity_fold=PARITY_FOLD, design=CONFIG["design"], mode=CONFIG["mode"])

Xtr = build_X(featurize(train_text), find_groups(train_text), CONFIG["mode"])
g_test = find_groups(test_text)
Xte = build_X(featurize(test_text), g_test, CONFIG["mode"])
stage("features", columns=Xtr.shape[1], test_groups=len(np.unique(g_test)),
      test_multi_groups=int((pd.Series(g_test).value_counts() >= 2).sum()))

if CONFIG["design"] == "flat":
    cfg = CONFIG["flat"]
    m = make_model(cfg["kind"], cfg["params"], n_jobs=N_JOBS)
    m.fit(Xtr, np.array([classes.index(v) for v in models]))
    assert list(m.classes_) == list(range(18))
    P18 = np.clip(m.predict_proba(Xte), 1e-6, 1)
    P18 /= P18.sum(1, keepdims=True)
else:
    fitted = {n: fit_node(n, c, Xtr, models, classes, n_jobs=N_JOBS) for n, c in CONFIG["nodes"].items()}
    P_top = predict_node(fitted["top"], Xte)
    P_sub = {n: predict_node(f, Xte) for n, f in fitted.items() if n != "top"}
    P18 = combine(P_top, P_sub, classes, CONFIG["use_cg"])
assert P18.shape == (len(test), 18) and np.isfinite(P18).all()
stage("trees", rows=len(test))

z = post(np.log(np.clip(P18, 1e-12, None)), g_test)
pred = np.array(classes)[z.argmax(1)]
sub = pd.DataFrame({"id": test.id.astype(str).values, "model": pred})
assert len(sub) == len(test) and sub.id.is_unique and sub.model.isin(classes).all()
sub.to_parquet(f"{OUT_DIR}/submission.parquet", index=False)
vc = sub.model.value_counts()
SUMMARY["prediction_counts"] = vc.to_dict()
if PARITY_FOLD >= 0:
    from sklearn.metrics import balanced_accuracy_score
    SUMMARY["parity_score"] = round(float(balanced_accuracy_score(test.model, pred)), 4)
    print("PARITY balanced accuracy", SUMMARY["parity_score"], flush=True)
stage("written", rows=len(sub), classes_predicted=len(vc), top_share=round(float(vc.iloc[0] / len(sub)), 3),
      post_applied=len(test) >= MIN_POST_ROWS)
print(vc.to_string(), "\nwritten", len(sub), "rows", flush=True)
