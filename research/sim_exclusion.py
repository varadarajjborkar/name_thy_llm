"""Soft "one passage per model per prompt" constraint, judged on the calibrated hidden-test simulator.

Why: in train.csv every prompt is answered once by each model; hard Hungarian assignment inside found clusters gave
+10 OOF but -3.5 on the LB (probe P1), because clusters of N/18 merge ~4.5 real prompts (the simulator reproduces
that -3.5). The hidden test has ~4 passages per prompt, so the constraint may still carry information inside
small, purer clusters, if applied softly.
Mean-field exclusion: within a found cluster, passage i's evidence for model c is reduced by how likely the other
members are to be c:  z_i(c) += lam * sum_{j != i, j in cluster} log(1 - p_j(c) + eps)
(the log-probability that none of the others is c, if they were independent), p = softmax of the v9-centred blend.
Applied after v9's centering, before partial sinkhorn 0.45. Grid: cluster size for exclusion in {3, 4, 6, 9}
(found separately from centering's N/9 clusters), lam in {0.25 .. 3.0}, one pass (a smoke test showed two passes never better).
Cells with posterior >= 0.01 (logs/test_sim_v2_cells.csv), 30 draws each, common random numbers.
Decision rule: adopt only if >= +0.75 vs v9 (selection over 24 settings) and never worse in a plausible cell.
Run: .venv/Scripts/python.exe research/sim_exclusion.py
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.metrics import balanced_accuracy_score as bacc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import test_sim as T  # noqa: E402

LOG = ROOT / "logs/sim_exclusion.log"
SIZES, LAMS, PASSES = [3, 4, 6, 9], [0.25, 0.5, 1.0, 1.5, 2.0, 3.0], [1]


def say(*x):
    msg = " ".join(str(v) for v in x)
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def exclude(lb, groups, lam, passes):
    """Mean-field exclusion inside clusters; lb are log-probabilities (rows sum to 1 in prob space)."""
    z = lb.copy()
    for _ in range(passes):
        p = np.exp(T.log_softmax(z))
        lq = np.log(np.clip(1 - p, 1e-6, 1))
        out = lb.copy()
        for g in np.unique(groups):
            m = np.flatnonzero(groups == g)
            if len(m) >= 2:
                tot = lq[m].sum(0)
                out[m] += lam * (tot[None, :] - lq[m])  # sum over the other members only
        z = out
    return T.log_softmax(z)


def one(m, alpha, i):
    keep = T.draw(m, alpha, np.random.default_rng(10_000 + i))
    yy, v7 = T.y[keep], T.PIPE["v7"][keep]
    g9 = T.find_groups(T.text[keep], 9)
    lb = T.log_softmax(T.center(v7, g9, 0.25))  # v9 centering
    fin = lambda z: ((1 - 0.45) * z + 0.45 * T.sinkhorn(z)).argmax(1)  # noqa: E731  v9 sinkhorn
    out = {"m": m, "alpha": alpha, "i": i, "v9": 100 * bacc(yy, fin(lb))}
    for s in SIZES:
        gs = T.find_groups(T.text[keep], s)
        for lam in LAMS:
            for ps in PASSES:
                out[f"s{s}_l{lam}_p{ps}"] = 100 * bacc(yy, fin(exclude(lb, gs, lam, ps)))
    return out


if __name__ == "__main__":
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    LOG.write_text("", encoding="utf-8")
    C = pd.read_csv(ROOT / "logs/test_sim_v2_cells.csv")
    C = C[C.post_prob >= 0.01].copy()
    C["post_prob"] /= C.post_prob.sum()
    say(f"{len(C)} cells x {T.D} draws, 2 workers")
    R = pd.DataFrame(Parallel(n_jobs=2, batch_size=4)(delayed(one)(r.m, r.alpha, i)
                                                       for r in C.itertuples() for i in range(T.D)))
    R.to_parquet(ROOT / "logs/sim_exclusion_draws.parquet", index=False)
    full = one(18, np.inf, 0)  # full train.csv structure (all 18 per prompt): the OOF view, for reference
    R = R.merge(C[["m", "alpha", "post_prob"]], on=["m", "alpha"])
    R["wt"] = R.post_prob / T.D
    plaus = C[C.post_prob >= 0.05]
    res = []
    for k in [c for c in R.columns if c.startswith("s") and "_l" in c]:
        diff = R[k] - R.v9
        dm = float((R.wt * diff).sum())
        se = float(np.sqrt((R.wt ** 2 * (diff - dm) ** 2).sum()))
        worst = min(float(diff[(R.m == p.m) & (R.alpha == p.alpha)].mean()) for p in plaus.itertuples())
        res.append({"setting": k, "vs_v9": dm, "se": se, "worst_plausible_cell": worst,
                    "full_structure_vs_v9": full[k] - full["v9"]})
    S = pd.DataFrame(res).sort_values("vs_v9", ascending=False)
    S.to_csv(ROOT / "logs/sim_exclusion.csv", index=False)
    say(f"v9 expected {(R.wt * R.v9).sum():.2f}; exclusion settings vs v9 (posterior-weighted):")
    say(S.round(3).to_string(index=False))
    ok = S[(S.vs_v9 >= 0.75) & (S.worst_plausible_cell >= 0)]
    say("DECISION: " + (f"adopt {ok.iloc[0].setting} (+{ok.iloc[0].vs_v9:.2f})" if len(ok) else "keep v9"))
