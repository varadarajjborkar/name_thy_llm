"""Judge member changes on the calibrated hidden-test simulator (not on full-structure OOF, which over-rates them).
Posterior over (m, alpha) from logs/test_sim_v2_cells.csv (paired LB probes P1, P2, P3, P5; research/test_sim_v2.py).
Each candidate member set is scored with the v9 post (centering 0.25, clusters N/9, partial sinkhorn 0.45) on the
same simulated tests (common random numbers, seed 10_000 + i, cells with posterior >= 0.01, 30 draws each), and
reported as the posterior-weighted difference to v9's members, with SE and the worst plausible cell.
Candidates are fixed in advance (few, to avoid selection bias):
  v9        v7 members (s4 0.5, s2mix 0.3, svm 0.1, lgb 0.1, atlas_pool 0.15)
  s4x2      s4 replaced by the mean log-probabilities of the s4 seeds (42 and 7), same weights
Also prints the full-structure OOF numbers for comparison. Caveat: the P4 diagnostic says member gains have shrunk
on the hidden test before (v7 vs v5); the simulator cannot see that shrinkage, so treat member gains as upper bounds.
Run: .venv/Scripts/python.exe research/sim_members.py
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

LOG = ROOT / "logs/sim_members.log"


def say(*x):
    msg = " ".join(str(v) for v in x)
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def family(names):
    """Mean log-softmax of several out-of-fold families, aligned to the blend rows (T.ids order)."""
    out = []
    for runs in names:
        parts = []
        for k, r in enumerate(runs):
            d = ROOT / "checkpoints" / r
            vi = pd.read_csv(d / "val_ids.csv").id.astype(str)
            parts.append(pd.DataFrame(T.log_softmax(np.load(d / "val_logits.npy")), index=vi))
        f = pd.concat(parts)
        assert f.index.is_unique and set(f.index) == set(T.ids)
        out.append(f.loc[T.ids].to_numpy())
    return sum(out) / len(out)


d, tp = T.d, T.tp
s4_42 = family([[f"s4ff{k}" for k in range(4)]])
assert np.allclose(s4_42, d["s4"]), "s4 family differs from blend_parts"
s4x2 = family([[f"s4ff{k}" for k in range(4)], [f"s4ff{k}s7" for k in range(4)]])
rest = 0.3 * d["s2mix"] + 0.1 * d["svm"] + 0.1 * d["lgb"] + 0.15 * tp["atlas_pool"]
CAND = {"v9": 0.5 * d["s4"] + rest, "s4x2": 0.5 * s4x2 + rest}


def one(m, alpha, i):
    keep = T.draw(m, alpha, np.random.default_rng(10_000 + i))
    yy, g = T.y[keep], T.find_groups(T.text[keep], 9)
    return {"m": m, "alpha": alpha, "i": i,
            **{k: 100 * bacc(yy, T.post(z[keep], g, 0.25, 0.45).argmax(1)) for k, z in CAND.items()}}


if __name__ == "__main__":
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    LOG.write_text("", encoding="utf-8")
    y = T.y
    say("full-structure OOF (raw argmax): s4 seed 42 %.4f | seed 7 %.4f | mean of both %.4f"
        % (bacc(y, s4_42.argmax(1)), bacc(y, family([[f"s4ff{k}s7" for k in range(4)]]).argmax(1)),
           bacc(y, s4x2.argmax(1))))
    for k, z in CAND.items():
        say(f"full-structure OOF {k}: raw {bacc(y, z.argmax(1)):.4f}")
    C = pd.read_csv(ROOT / "logs/test_sim_v2_cells.csv")
    C = C[C.post_prob >= 0.01].copy()
    C["post_prob"] /= C.post_prob.sum()
    R = pd.DataFrame(Parallel(n_jobs=2, batch_size=4)(delayed(one)(r.m, r.alpha, i)
                                                       for r in C.itertuples() for i in range(T.D)))
    R = R.merge(C[["m", "alpha", "post_prob"]], on=["m", "alpha"])
    R["wt"] = R.post_prob / T.D
    plaus = C[C.post_prob >= 0.05]
    for k in CAND:
        diff = R[k] - R["v9"]
        dm = float((R.wt * diff).sum())
        se = float(np.sqrt((R.wt ** 2 * (diff - dm) ** 2).sum()))
        worst = min(float(diff[(R.m == p.m) & (R.alpha == p.alpha)].mean()) for p in plaus.itertuples())
        say(f"sim (v9 post) {k:6s} expected {(R.wt * R[k]).sum():.2f}  vs v9 {dm:+.2f} (SE {se:.2f})  "
            f"worst plausible cell {worst:+.2f}")
