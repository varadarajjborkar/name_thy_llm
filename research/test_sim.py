"""Calibrated simulator of the hidden test, fitted to our own paired leaderboard probes, then used to choose the
post-processing (prompt centering weight, sinkhorn strength, cluster count) for v7.

Why: our out-of-fold score (every prompt answered by all 18 models, equal mix) over-rates post-processing. Kaggle says
the hidden test has unseen prompts, "not all 18" answers per prompt, and an unequal model mix. Two unknowns:
  m      rate parameter for passages per prompt (the REALISED mean, rows / prompts, is reported separately: at least
         one passage per prompt is forced and inclusion probabilities are capped at 1)
  alpha  unevenness of the model mix (Dirichlet concentration; inf = equal mix)
Generative model per simulated test (the 120 prompts of folds 0-3, out-of-fold predictions):
  w ~ Dirichlet(alpha * 1_18); model j answers prompt p with probability min(1, m * w_j) (at least one per prompt).
  Prompt groups are found blind as the notebook does (TF-IDF content words, SVD, average-link cosine, round(N/18)
  clusters), so the clusters-per-prompt ratio matches the real ~4000-row test. Known limit (issue 93): the absolute
  scale does not (200-1000 simulated rows from 120 reused prompts vs ~4000 rows of ~1000 prompts; max_df = 0.05
  therefore cuts at a different document count), so group purity may differ from the real test.
Paired LB probes (same members, only the post step changed; each difference cancels the members' domain shift):
  P1 TF-IDF blend + group assignment   0.367 vs 0.402 raw   -> -3.5
  P2 TF-IDF blend + full sinkhorn      0.398 vs 0.402 raw   -> -0.4
  P3 3-family Qwen blend: + centering 0.5 (both with full sinkhorn)  0.572 vs 0.555 -> +1.7
Likelihood per cell: chi2 = sum_k (mu_k - lb_k)^2 / (sigma_k^2 + sd_k^2 / D), where sigma_k is the LB sampling error
of a paired difference on ~1600 public rows, 100 * sqrt(f_k / 1600) with f_k the simulated fraction of rows whose
correctness flips between the two variants, and sd_k^2 / D the simulation's own Monte Carlo variance. Flat prior.
Common random numbers: draw i uses the same seed in every cell. D draws per cell, 4 single-thread workers.
Stage 2: for every draw, v7 is scored under 60 post settings; posterior-weighted means and paired differences
against the shipped setting (centering 0.5, sinkhorn 0.45, clusters N/18) are reported.
Run: .venv/Scripts/python.exe research/test_sim.py
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import time  # noqa: E402
from itertools import product  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.cluster import AgglomerativeClustering  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import balanced_accuracy_score as bacc  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs/test_sim.log"
T0 = time.time()


def say(*x):
    msg = " ".join(str(v) for v in x)
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def sinkhorn(lp, iters=100):  # as in the notebook
    p = np.exp(lp - lp.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    for _ in range(iters):
        p /= p.sum(0, keepdims=True)
        p /= p.sum(1, keepdims=True)
    return np.log(p + 1e-12)


def find_groups(texts, n_per_group=18):
    """The notebook's prompt_centering grouping (empty TF-IDF rows become singletons)."""
    Xg = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, max_df=0.05).fit_transform(texts)
    ok = np.asarray(Xg.getnnz(axis=1) > 0)
    groups = np.arange(len(texts)) + len(texts)
    if ok.sum() >= 2 * 18:
        Zg = normalize(TruncatedSVD(min(256, Xg.shape[1] - 1, int(ok.sum()) - 1), random_state=0)
                       .fit_transform(Xg[ok]))
        groups[ok] = AgglomerativeClustering(n_clusters=max(1, round(ok.sum() / n_per_group)), metric="cosine",
                                             linkage="average").fit_predict(Zg)
    return groups


def center(z, groups, w):
    lb = log_softmax(z)
    out = z.copy()
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= 2:
            out[m] = lb[m] - w * lb[m].mean(0)
    return out


def post(z, groups, w_center, a_sink):
    z = center(z, groups, w_center) if w_center else z
    lb = log_softmax(z)
    return (1 - a_sink) * lb + a_sink * sinkhorn(lb) if a_sink else lb


def assign_groups(lp, groups, n_cls=18):  # research/assign.py, as submitted for P1
    lp = log_softmax(lp)
    pred = lp.argmax(1).copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        reps = int(np.ceil(len(idx) / n_cls))
        rows, cols = linear_sum_assignment(-np.tile(lp[idx], (1, reps)))
        pred[idx[rows]] = cols % n_cls
    return pred


# ---------------------------------------------------------------- out-of-fold members (folds 0-3)
train = pd.read_csv(ROOT / "data/train.csv")
train["id"] = train.id.astype(str)
d = np.load(ROOT / "data/processed/blend_parts.npz", allow_pickle=True)
ids, y = [str(i) for i in d["ids"]], d["y"]
tp = np.load(ROOT / "data/processed/trainonly_parts.npz", allow_pickle=True)
assert [str(i) for i in tp["ids"]] == ids and (tp["y"] == y).all()
row = pd.Series(np.arange(len(train)), index=train.id).loc[ids].to_numpy()
text = train.text.fillna("").astype(str).to_numpy()[row]
prompt = train.prompt_id.to_numpy()[row]
PIPE = {
    "tfidf": 0.5 * d["svm"] + 0.5 * d["lgb"],                       # LB 0.402 raw
    "q3": 0.8 * d["s2mix"] + 0.1 * d["svm"] + 0.1 * d["lgb"],       # LB 0.555 full sinkhorn / 0.572 + centering
    "v7": 0.5 * d["s4"] + 0.3 * d["s2mix"] + 0.1 * d["svm"] + 0.1 * d["lgb"] + 0.15 * tp["atlas_pool"],  # LB 0.571
}
LB = {"P1_assign": -3.5, "P2_sink": -0.4, "P3_center": +1.7}
N_PUBLIC = 1600
PROMPTS = np.unique(prompt)
D = 30
GRID_M = [1.5, 2, 3, 4, 6, 9, 12]
GRID_A = [0.5, 1.0, 2.0, 5.0, np.inf]
POST_W, POST_A, POST_G = [0, 0.25, 0.5, 0.75], [0, 0.2, 0.45, 0.7, 1.0], [18, 9, 5]
SHIPPED = "w0.5_a0.45_g18"


def draw(m, alpha, rng):
    w = np.full(18, 1 / 18) if np.isinf(alpha) else rng.dirichlet(np.full(18, alpha))
    pj = np.minimum(1.0, m * w)
    keep = []
    for p in PROMPTS:
        r = np.flatnonzero(prompt == p)  # 18 rows, one per model
        inc = rng.random(len(r)) < pj[y[r]]
        if not inc.any():
            inc[rng.choice(len(r), p=w[y[r]] / w[y[r]].sum())] = True
        keep += list(r[inc])
    return np.array(sorted(keep))


def simulate(m, alpha, i):
    keep = draw(m, alpha, np.random.default_rng(10_000 + i))  # common random numbers across cells
    yy = y[keep]
    G = {g: find_groups(text[keep], g) for g in POST_G}
    tf, q3, v7 = (PIPE[k][keep] for k in ("tfidf", "q3", "v7"))
    ok = lambda z: z.argmax(1) == yy  # noqa: E731
    s = lambda pred: 100 * bacc(yy, pred)  # noqa: E731
    flip = lambda a, b: float(np.mean(a != b))  # noqa: E731
    tf_raw, tf_as, tf_sk = tf.argmax(1), assign_groups(tf, G[18]), post(tf, G[18], 0, 1.0).argmax(1)
    q3_s, q3_c = post(q3, G[18], 0, 1.0).argmax(1), post(q3, G[18], 0.5, 1.0).argmax(1)
    out = {"m": m, "alpha": alpha, "i": i, "rows": len(keep), "per_prompt": len(keep) / len(PROMPTS),
           "P1_assign": s(tf_as) - s(tf_raw), "P2_sink": s(tf_sk) - s(tf_raw), "P3_center": s(q3_c) - s(q3_s),
           "f_P1_assign": flip(tf_as == yy, tf_raw == yy), "f_P2_sink": flip(tf_sk == yy, tf_raw == yy),
           "f_P3_center": flip(q3_c == yy, q3_s == yy)}
    for g in POST_G:
        for w in POST_W:
            lb = log_softmax(center(v7, G[g], w) if w else v7)
            sk = sinkhorn(lb)
            for a in POST_A:
                out[f"w{w}_a{a}_g{g}"] = s(((1 - a) * lb + a * sk).argmax(1))
    del ok
    return out


if __name__ == "__main__":
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)  # loky workers inherit the priority class
    except Exception:
        pass
    LOG.write_text("", encoding="utf-8")
    jobs = [(m, a, i) for m, a in product(GRID_M, GRID_A) for i in range(D)]
    say(f"{len(jobs)} simulated tests ({len(GRID_M) * len(GRID_A)} cells x {D} draws), 4 workers")
    R = pd.DataFrame(Parallel(n_jobs=4, batch_size=4)(delayed(simulate)(*j) for j in jobs))
    R.to_parquet(ROOT / "logs/test_sim_draws.parquet", index=False)
    say(f"simulated in {time.time() - T0:.0f}s")
    cells = []
    for (m, a), g in R.groupby(["m", "alpha"]):
        c = {"m": m, "alpha": a, "per_prompt": g.per_prompt.mean(), "chi2": 0.0}
        for k in LB:
            sig2 = (100 * np.sqrt(g[f"f_{k}"].mean() / N_PUBLIC)) ** 2 + g[k].var() / len(g)
            c[k], c[f"se_{k}"] = g[k].mean(), np.sqrt(sig2)
            c["chi2"] += (g[k].mean() - LB[k]) ** 2 / sig2
        cells.append(c)
    C = pd.DataFrame(cells)
    C["post_prob"] = np.exp(-(C.chi2 - C.chi2.min()) / 2)
    C["post_prob"] /= C.post_prob.sum()
    say("\ncell fit (sorted by chi2): sim probe means vs LB P1 -3.5 / P2 -0.4 / P3 +1.7")
    say(C.sort_values("chi2").round(3).to_string(index=False))
    marg_m = C.groupby("m").post_prob.sum()
    marg_a = C.groupby("alpha").post_prob.sum()
    say("\nposterior over m: " + ", ".join(f"{k}: {v:.2f}" for k, v in marg_m.items()))
    say("posterior over alpha: " + ", ".join(f"{k}: {v:.2f}" for k, v in marg_a.items()))
    say(f"posterior mean passages per prompt: {(C.per_prompt * C.post_prob).sum():.2f}")
    # stage 2: v7 post settings under the posterior
    keys = [c for c in R.columns if c.startswith("w")]
    W = R.merge(C[["m", "alpha", "post_prob"]], on=["m", "alpha"])
    W["wt"] = W.post_prob / D
    res = []
    for k in keys:
        diff = W[k] - W[SHIPPED]
        mean = float((W.wt * W[k]).sum())
        dm = float((W.wt * diff).sum())
        se = float(np.sqrt((W.wt ** 2 * (diff - dm) ** 2).sum()))  # weighted SE over draws (cells independent)
        plaus = C[C.post_prob >= 0.05][["m", "alpha"]]
        worst = min(float(diff[(W.m == r.m) & (W.alpha == r.alpha)].mean()) for r in plaus.itertuples())
        res.append({"setting": k, "expected": mean, "vs_shipped": dm, "se": se, "worst_plausible_cell": worst})
    S = pd.DataFrame(res).sort_values("expected", ascending=False)
    S.to_csv(ROOT / "logs/test_sim_post.csv", index=False)
    say(f"\nv7 post settings, posterior-weighted (shipped {SHIPPED}); worst = mean diff in the worst cell with"
        " posterior >= 0.05")
    say(S.head(15).round(3).to_string(index=False))
    say("\nreference rows:")
    say(S[S.setting.isin([SHIPPED, "w0_a0_g18", "w0.5_a0_g18", "w0_a0.45_g18", "w0.5_a1.0_g18"])]
        .round(3).to_string(index=False))
    say(f"\ntotal {time.time() - T0:.0f}s")
