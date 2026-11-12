"""tree2 options, judged under a test-like distribution shift (Geirhos et al. 2020, "Shortcut Learning"): train.csv
has every prompt answered by all 18 models, but the hidden test has "only a few answers" per prompt (mizuno0522's
notebook; Kaggle Overview: not all 18) and an unequal model mix. tree1 (centered-only features) scored 0.376 in CV
and 0.263 on the LB.

Scenarios for each scored fold (folds 0-3, GroupKFold(5) by prompt_id; fold 4 untouched):
  full    all texts of the fold (the old, optimistic view)
  sparse  per prompt keep k ~ U{2..6} texts, models drawn with Dirichlet(1) weights (uneven mix), 3 draws averaged;
          prompt groups are then found on that subset only, as on the real test
Feature centering for TRAINING rows: "full" (groups of ~18) or "sparse" (each found group split at random into
chunks of 2-6, so training sees the same kind of centering the test will get).
Options: tree on style / style + next-word averages (ATLAS) / + next-word signal, feature mode raw / both;
letter-chunk TF-IDF + logistic regression (no centering, robust to groups); blends. Scores: model and company
balanced accuracy, with and without the notebook post-processing.
Run: .venv/Scripts/python.exe research/tree2_eval.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "features"), str(ROOT / "tree")]
import ntl_tree as T  # noqa: E402
from atlas_feats import COLS as ATLAS_COLS  # noqa: E402
from signal_feats import SIG_NAMES  # noqa: E402

try:
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass
LOG = ROOT / "logs/tree2_eval.log"


def say(*x):
    msg = " ".join(str(v) for v in x)
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


train = pd.read_csv(ROOT / "data/train.csv")
train["id"] = train.id.astype(str)
texts = train.text.fillna("").astype(str).to_numpy()
models = train.model.to_numpy()
CL = sorted(train.model.unique())
y = np.array([CL.index(m) for m in models])
yf = np.array([T.FAMILIES.index(T.fam_of(m)) for m in models])
fold = np.empty(len(train), dtype=int)
for k, (_, va) in enumerate(GroupKFold(5).split(train, train.model, train.prompt_id)):
    fold[va] = k
S = T.featurize(texts)
A = pd.read_parquet(ROOT / "data/processed/atlas_host_fp16.parquet").assign(id=lambda d: d.id.astype(str))
A = A.set_index("id").loc[train.id, ATLAS_COLS].to_numpy(np.float64)
G = pd.read_parquet(ROOT / "data/processed/signal_train.parquet").assign(id=lambda d: d.id.astype(str))
G = G.set_index("id").loc[train.id, SIG_NAMES].to_numpy(np.float64)
FEATS = {"style": S, "style+nextword": np.hstack([S, A]), "style+nextword+signal": np.hstack([S, A, G])}


def sparse_subset(rows, seed):
    """Test-like subset of the scored rows: 2-6 texts per prompt, uneven model mix."""
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(18))
    keep = []
    for p in np.unique(train.prompt_id.to_numpy()[rows]):
        r = rows[train.prompt_id.to_numpy()[rows] == p]
        k = min(len(r), rng.integers(2, 7))
        pr = w[y[r]] / w[y[r]].sum()
        keep += list(rng.choice(r, size=k, replace=False, p=pr))
    return np.array(sorted(keep))


def chunk_groups(groups, seed):
    """Split every found group at random into chunks of 2-6 (sparse centering for training rows)."""
    rng = np.random.default_rng(seed)
    out = np.empty_like(groups)
    nxt = 0
    for g in np.unique(groups):
        idx = rng.permutation(np.flatnonzero(groups == g))
        i = 0
        while i < len(idx):
            k = int(rng.integers(2, 7))
            out[idx[i:i + k]] = nxt
            nxt += 1
            i += k
    return out


SCEN = {}  # (fold, scenario) -> scored rows
for k in range(4):
    va = np.flatnonzero(fold == k)
    SCEN[(k, "full")] = va
    for s in range(3):
        SCEN[(k, f"sparse{s}")] = sparse_subset(va, 100 * k + s)
say("==== tree2_eval: sparse subsets per fold have", [len(SCEN[(0, f'sparse{s}')]) for s in range(3)], "rows (fold 0)")
GROUPS = {key: T.find_groups(texts[r].tolist()) for key, r in SCEN.items()}
TRG = {k: T.find_groups(texts[fold != k].tolist()) for k in range(4)}


def tree_probs(feat, mode, train_centering):
    """dict scenario-key -> 18-class probabilities of a flat LightGBM (3 seeds)."""
    M = FEATS[feat]
    out = {}
    for k in range(4):
        tr = np.flatnonzero(fold != k)
        g = TRG[k] if train_centering == "full" else chunk_groups(TRG[k], k)
        Xtr = T.build_X(M[tr], g, mode)
        m = T.make_model("lgbm", dict(n=300, lr=0.05, leaves=15, min_child=20, ff=0.5, l2=1), n_jobs=6)
        m.fit(Xtr, y[tr])
        for key, r in SCEN.items():
            if key[0] == k:
                p = np.clip(m.predict_proba(T.build_X(M[r], GROUPS[key], mode)), 1e-6, 1)
                out[key] = p / p.sum(1, keepdims=True)
    return out


def chars_probs(C=2.0):
    out = {}
    for k in range(4):
        tr = np.flatnonzero(fold != k)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=60000)
        Xtr = vec.fit_transform(texts[tr])
        lr = LogisticRegression(C=C, max_iter=3000).fit(Xtr, y[tr])
        for key, r in SCEN.items():
            if key[0] == k:
                out[key] = lr.predict_proba(vec.transform(texts[r]))
    return out


def report(name, P):
    res = {}
    for scen in ["full", "sparse"]:
        keys = [key for key in SCEN if (key[1] == "full") == (scen == "full")]
        raw, post, fam = [], [], []
        for key in keys:
            r = SCEN[key]
            lp = np.log(np.clip(P[key], 1e-12, None))
            pr, pp = lp.argmax(1), T.post(lp, GROUPS[key]).argmax(1)
            raw.append(bacc(y[r], pr))
            post.append(bacc(y[r], pp))
            fam.append(bacc(yf[r], [T.FAMILIES.index(T.fam_of(CL[i])) for i in pr]))
        res[scen] = (np.mean(raw), np.mean(post), np.mean(fam))
    say(f"  {name:48s} FULL raw {res['full'][0]:.3f} post {res['full'][1]:.3f} company {res['full'][2]:.3f} | "
        f"SPARSE raw {res['sparse'][0]:.3f} post {res['sparse'][1]:.3f} company {res['sparse'][2]:.3f}")
    return res


R = {}
say("model / company balanced accuracy; FULL = all 18 per prompt (old view), SPARSE = 2-6 per prompt, uneven mix")
for feat in FEATS:
    for mode, trc in [("raw", "full"), ("centered", "full"), ("both", "full"), ("both", "sparse"),
                      ("centered", "sparse")]:
        name = f"tree {feat} {mode} (train centering {trc})"
        R[name] = tree_probs(feat, mode, trc)
        report(name, R[name])
R["letters LR"] = chars_probs()
report("letter chunks + logistic regression", R["letters LR"])


def blend(a, b, wa):
    return {key: np.exp(wa * np.log(np.clip(R[a][key], 1e-12, None)) + (1 - wa) * np.log(np.clip(R[b][key], 1e-12,
                                                                                                None)))
            for key in SCEN}


best_tree = max((n for n in R if n.startswith("tree")),
                key=lambda n: np.mean([bacc(y[SCEN[k]], R[n][k].argmax(1)) for k in SCEN if k[1] != "full"]))
say(f"best tree under SPARSE (raw): {best_tree}")
for wa in [0.3, 0.5, 0.7]:
    report(f"blend {wa} x best tree + {1 - wa:.1f} x letters", blend(best_tree, "letters LR", wa))
