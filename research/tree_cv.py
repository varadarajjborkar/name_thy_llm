"""Cross-validation and knob search for the feature-tree pipeline (tree/ntl_tree.py). train.csv only.

Protocol: GroupKFold(5) by prompt_id, as everywhere in this project. Folds 0-3 are scored out of fold; fold 4 is the
lockbox: it is never SCORED except by `--lockbox` (one final check, after every choice is made). Its rows do sit in the
training part of folds 0-3, the same convention as finetune/train.py --cv-fold, so no choice ever looks at its score.
Prompt groups for feature centering are found WITHOUT labels, separately on the training part and on the scored part,
exactly as they would be found on the hidden test set.

Stages
  A  feature mode: raw / centered / both, with one default flat LightGBM (is prompt centering of features useful?)
  B  every node (top = company, claude, gemini, gpt, glm, cg = Claude+GLM): random search over LightGBM (boosting),
     RandomForest and ExtraTrees (bagging); the best of each kind by the node's own out-of-fold log loss.
     Top node: inverse-frequency weight alpha in {0, 0.5, 1}.
  C  end to end: company-first (with / without the Claude+GLM tree, each alpha) against a flat 18-class tree tuned
     with the same budget. Scores: company balanced accuracy, model balanced accuracy (raw argmax and after the
     notebook's post-processing).
Writes tree/best_config.json, data/processed/tree_oof.npz, log logs/tree_cv.log.
Run: .venv/Scripts/python.exe research/tree_cv.py            (search, folds 0-3)
     .venv/Scripts/python.exe research/tree_cv.py --lockbox  (one-time check of the chosen config on fold 4)
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "features"), str(ROOT / "tree")]
import ntl_tree as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lockbox", action="store_true")
ap.add_argument("--n-jobs", type=int, default=6)  # leave CPU for the user's other programs
ap.add_argument("--quick", action="store_true", help="tiny search, for a smoke test of the code only")
a = ap.parse_args()

try:  # below-normal priority: the user's other project and normal programs come first
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

LOG = ROOT / "logs/tree_cv.log"
t0 = time.time()


def say(*x):
    msg = f"{time.strftime('%H:%M:%S')} [{time.time() - t0:6.0f}s] " + " ".join(str(v) for v in x)
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


train = pd.read_csv(ROOT / "data/train.csv")
train["text"] = train.text.fillna("").astype(str)
texts, models = train.text.to_numpy(), train.model.to_numpy()
CLASSES = sorted(train.model.unique())
y18 = np.array([CLASSES.index(m) for m in models])
yfam = np.array([T.FAMILIES.index(T.fam_of(m)) for m in models])
fold = np.empty(len(train), dtype=int)
for k, (_, va) in enumerate(GroupKFold(5).split(train, train.model, train.prompt_id)):
    fold[va] = k
VAL_FOLDS = [4] if a.lockbox else [0, 1, 2, 3]
NODES = ["top", "claude", "gemini", "gpt", "glm", "cg"]

say(f"==== tree_cv start ({'LOCKBOX' if a.lockbox else 'search'}), {len(train)} texts, n_jobs {a.n_jobs}")
F = T.featurize(texts)
say(f"features: {F.shape[1]} style columns (word lists and sentence starters excluded)")

# prompt groups, found without labels on each side of every split (as on the hidden test)
SPLITS = {}
for k in VAL_FOLDS:
    tr = np.flatnonzero(fold != k) if not a.lockbox else np.flatnonzero(fold != 4)
    va = np.flatnonzero(fold == k)
    SPLITS[k] = (tr, va, T.find_groups(texts[tr].tolist()), T.find_groups(texts[va].tolist()))
    g = SPLITS[k][3]
    pur = pd.crosstab(g, train.prompt_id.to_numpy()[va]).max(axis=1).sum() / len(va)
    say(f"fold {k}: train {len(tr)} / scored {len(va)} | scored-side prompt groups {len(np.unique(g))}, purity {pur:.3f}")


def Xs(k, mode):
    tr, va, gtr, gva = SPLITS[k]
    return T.build_X(F[tr], gtr, mode), T.build_X(F[va], gva, mode)


def oof_node(node, cfg, mode):
    """Out-of-fold probabilities of one node for every scored row (all rows, not only the node's own)."""
    out = {}
    for k in VAL_FOLDS:
        tr, va, _, _ = SPLITS[k]
        Xtr, Xva = Xs(k, mode)
        fitted = T.fit_node(node, cfg, Xtr, models[tr], CLASSES, n_jobs=a.n_jobs)
        out[k] = T.predict_node(fitted, Xva)
    return np.vstack([out[k] for k in VAL_FOLDS])


VA = np.concatenate([SPLITS[k][1] for k in VAL_FOLDS])        # scored rows, in fold order
GVA = np.concatenate([SPLITS[k][3] + 100_000 * k for k in VAL_FOLDS])  # scored-side groups, unique across folds
FOLD_VA = fold[VA]


def node_score(node, P):
    rows = T.node_rows(node, models[VA])
    cl = T.node_classes(node, CLASSES)
    y = np.array([cl.index(v) for v in T.node_target(node, models[VA][rows])])
    p = P[rows] / P[rows].sum(1, keepdims=True)
    return log_loss(y, p, labels=list(range(len(cl)))), bacc(y, p.argmax(1))


def scores(P18):
    """company balanced acc, model balanced acc raw, model balanced acc after post, per-fold post."""
    lp = np.log(np.clip(P18, 1e-12, None))
    pr = lp.argmax(1)
    post = np.empty_like(lp)
    for k in VAL_FOLDS:  # post-processing runs per scored fold, as on one test set
        m = FOLD_VA == k
        post[m] = T.post(lp[m], GVA[m])
    pp = post.argmax(1)
    fam_pred = np.array([T.FAMILIES.index(T.fam_of(CLASSES[i])) for i in pp])
    per_fold = [bacc(y18[VA][FOLD_VA == k], pp[FOLD_VA == k]) for k in VAL_FOLDS]
    return dict(company=bacc(yfam[VA], fam_pred), raw=bacc(y18[VA], pr), post=bacc(y18[VA], pp), folds=per_fold,
                pred=pp)


rng = np.random.default_rng(0)


def grid(kind, n):
    if kind == "lgbm":
        g = [dict(n=n_, lr=lr, leaves=lv, min_child=mc, ff=ff, l2=l2) for n_, lr, lv, mc, ff, l2 in itertools.product(
            [150, 300, 600], [0.03, 0.05, 0.1], [7, 15, 31], [5, 10, 20, 40], [0.3, 0.5, 0.8], [0, 1, 5])]
    else:
        g = [dict(n=500, mf=mf, leaf=lf, depth=d) for mf, lf, d in itertools.product(
            ["sqrt", 0.1, 0.25], [1, 2, 4], [None, 10])]
    idx = rng.choice(len(g), size=min(n, len(g)), replace=False)
    return [g[i] for i in idx]


N_CFG = {"lgbm": 2, "rf": 1, "et": 1} if a.quick else {"lgbm": 10, "rf": 6, "et": 6}
DEFAULT_LGBM = dict(n=300, lr=0.05, leaves=15, min_child=20, ff=0.5, l2=1)

def oof_flat(cfg, mode):
    """Out-of-fold probabilities of one flat 18-class tree."""
    out = {}
    for k in VAL_FOLDS:
        tr, va, _, _ = SPLITS[k]
        Xtr, Xva = Xs(k, mode)
        m = T.make_model(cfg["kind"], cfg["params"], n_jobs=a.n_jobs)
        m.fit(Xtr, y18[tr])
        assert list(m.classes_) == list(range(18))
        p = np.clip(m.predict_proba(Xva), 1e-6, 1)
        out[k] = p / p.sum(1, keepdims=True)
    return np.vstack([out[k] for k in VAL_FOLDS])


if a.lockbox:
    best = json.load(open(ROOT / "tree/best_config.json"))
    say("lockbox config:", json.dumps(best, default=str))
    mode = best["mode"]
    if best["design"] == "flat":
        P18 = oof_flat(best["flat"], mode)
    else:
        P_top = oof_node("top", best["nodes"]["top"], mode)
        P_sub = {n: oof_node(n, best["nodes"][n], mode) for n in NODES[1:]}
        P18 = T.combine(P_top, P_sub, CLASSES, best["use_cg"])
    s = scores(P18)
    say(f"LOCKBOX fold 4: company {s['company']:.4f} | model raw {s['raw']:.4f} | model after post {s['post']:.4f}")
    say(f"  (out-of-fold estimate on folds 0-3 was: company {best['cv']['company']:.4f} | post {best['cv']['post']:.4f})")
    sys.exit(0)

# ---------------------------------------------------------------- stage A: feature mode
say("== stage A: feature mode, flat 18-class LightGBM (default knobs)")
flat_cfg = {"kind": "lgbm", "params": DEFAULT_LGBM, "alpha": 0}

A = {}
for mode in ["raw", "centered", "both"]:
    s = scores(oof_flat(flat_cfg, mode))
    A[mode] = s
    say(f"  {mode:9s} company {s['company']:.4f} | model raw {s['raw']:.4f} | post {s['post']:.4f} | folds "
        + " ".join(f"{v:.3f}" for v in s["folds"]))
mode = max(A, key=lambda m: A[m]["post"])
say(f"  -> feature mode '{mode}'")

# ---------------------------------------------------------------- stage B: node search
say("== stage B: node search (log loss decides inside a node)")
CAND = {}  # node -> list of (cfg, P, logloss, bacc)
for node in NODES:
    CAND[node] = []
    for kind in ["lgbm", "rf", "et"]:
        for p in grid(kind, N_CFG[kind]):
            cfg = {"kind": kind, "params": p, "alpha": 0}
            P = oof_node(node, cfg, mode)
            ll, ba = node_score(node, P)
            CAND[node].append((cfg, P, ll, ba))
        best_k = min((c for c in CAND[node] if c[0]["kind"] == kind), key=lambda c: c[2])
        say(f"  {node:6s} {kind:4s} best log loss {best_k[2]:.4f} (node bal. acc {best_k[3]:.4f}) {best_k[0]['params']}")
    if node == "top":  # inverse-frequency weighting knob, on the best config of each kind
        for kind in ["lgbm", "rf", "et"]:
            base = min((c for c in CAND[node] if c[0]["kind"] == kind and c[0]["alpha"] == 0), key=lambda c: c[2])
            for alpha in [0.5, 1.0]:
                cfg = dict(base[0], alpha=alpha)
                P = oof_node(node, cfg, mode)
                ll, ba = node_score(node, P)
                CAND[node].append((cfg, P, ll, ba))
                say(f"  top    {kind:4s} alpha {alpha}: log loss {ll:.4f} | company bal. acc {ba:.4f}")

BEST = {n: min(CAND[n], key=lambda c: c[2]) for n in NODES[1:]}
for n in NODES[1:]:
    say(f"  chosen {n:6s}: {BEST[n][0]['kind']} (log loss {BEST[n][2]:.4f}, bal. acc inside {n} {BEST[n][3]:.4f})")

# ---------------------------------------------------------------- stage C: end to end
say("== stage C: end to end (folds 0-3, out of fold)")
RES = []
for alpha in [0, 0.5, 1.0]:
    top = min((c for c in CAND["top"] if c[0]["alpha"] == alpha), key=lambda c: c[2])
    for use_cg in [False, True]:
        s = scores(T.combine(top[1], {n: BEST[n][1] for n in NODES[1:]}, CLASSES, use_cg))
        RES.append((f"company-first alpha {alpha} {'+ Claude/GLM tree' if use_cg else ''}", s, alpha, use_cg, top))
say("  flat 18-class search (same budget) ...")
flat = []
for kind in ["lgbm", "rf", "et"]:
    for p in grid(kind, N_CFG[kind]):
        cfg = {"kind": kind, "params": p, "alpha": 0}
        P = oof_flat(cfg, mode)
        flat.append((cfg, P, log_loss(y18[VA], P / P.sum(1, keepdims=True), labels=list(range(18)))))
fb = min(flat, key=lambda c: c[2])
RES.append((f"flat 18-class {fb[0]['kind']}", scores(fb[1]), None, None, fb))
for name, s, *_ in RES:
    say(f"  {name:42s} company {s['company']:.4f} | model raw {s['raw']:.4f} | post {s['post']:.4f} | folds "
        + " ".join(f"{v:.3f}" for v in s["folds"]))

win = max(RES, key=lambda r: r[1]["post"])
name, s, alpha, use_cg, top = win
say(f"WINNER: {name}: model {s['post']:.4f} after post, company {s['company']:.4f}")
pp = s["pred"]
rec = pd.Series({c: (pp[y18[VA] == i] == i).mean() for i, c in enumerate(CLASSES)})
say("per-model recall (winner, after post):\n" + rec.round(3).to_string())
fam_true = np.array([T.FAMILIES[i] for i in yfam[VA]])
fam_pred = np.array([T.fam_of(CLASSES[i]) for i in pp])
say("company confusion (rows true, columns guessed):\n" + pd.crosstab(fam_true, fam_pred).to_string())

if alpha is None:
    best = {"design": "flat", "mode": mode, "flat": fb[0]}
else:
    best = {"design": "company-first", "mode": mode, "use_cg": use_cg,
            "nodes": {"top": top[0], **{n: BEST[n][0] for n in NODES[1:]}}}
best["cv"] = {"company": s["company"], "raw": s["raw"], "post": s["post"], "folds": s["folds"]}
best["stage_a"] = {m: {k: v for k, v in r.items() if k != "pred"} for m, r in A.items()}
if a.quick:  # a smoke test must never replace the real choice
    say("quick run: tree/best_config.json and tree_oof.npz NOT written")
    sys.exit(0)
(ROOT / "tree").mkdir(exist_ok=True)
json.dump(best, open(ROOT / "tree/best_config.json", "w"), indent=1, default=str)
np.savez(ROOT / "data/processed/tree_oof.npz", rows=VA, fold=FOLD_VA, pred=pp, y=y18[VA])
say("wrote tree/best_config.json and data/processed/tree_oof.npz")
