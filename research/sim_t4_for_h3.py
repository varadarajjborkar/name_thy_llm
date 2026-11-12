"""v9 with the hostlike-trained Qwen member (h3) replaced by the train.csv-only Qwen (t4: Qwen3-4B base, LoRA, stage 2
recipe of s4 but no external stage 1; checkpoints/t4f0-3, OOF 0.40, LB 0.434 alone).
Candidates, fixed in advance (weights sum 1.15 like v9, so the post sees the same scale):
  v9          0.5 s4 + 0.3 mean(s2, s3, h3) + 0.1 svm + 0.1 lgb + 0.15 atlas
  t4_for_h3   0.5 s4 + 0.3 mean(s2, s3, t4) + ...            (h3's seat given to t4)
  t4_added    v9 with 0.1 of the s2mix weight moved to t4: 0.2 mean(s2, s3, h3) + 0.1 t4 (h3 kept; reference)
Same simulator, posterior, draws, post and decision rule as research/sim_nohost.py (adopt only if >= +0.5 vs v9 and
never worse in a plausible cell). Run: .venv/Scripts/python.exe research/sim_t4_for_h3.py
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"  # before numpy is imported, so the cap applies to this process too
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.metrics import balanced_accuracy_score as bacc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import sim_nohost as N  # noqa: E402  (loads the simulator, the families s2/s3/h3 and the v9 base)
T = N.T

LOG = ROOT / "logs/sim_t4_for_h3.log"
t4 = N.family([ROOT / f"checkpoints/t4f{k}" for k in range(4)])
F = N.F
CAND = {"v9": N.CAND["v9"],
        "t4_for_h3": N.base + 0.3 * (F["s2"] + F["s3"] + t4) / 3,
        "t4_added": N.base + 0.2 * N.d["s2mix"] + 0.1 * t4}


def say(*x):
    msg = " ".join(str(v) for v in x)
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


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
    say(f"full-structure OOF raw: t4 alone {bacc(y, t4.argmax(1)):.4f}, h3 alone {bacc(y, F['h3'].argmax(1)):.4f}; "
        + ", ".join(f"{k} {bacc(y, z.argmax(1)):.4f}" for k, z in CAND.items()))
    C = pd.read_csv(ROOT / "logs/test_sim_v2_cells.csv")
    C = C[C.post_prob >= 0.01].copy()
    C["post_prob"] /= C.post_prob.sum()
    R = pd.DataFrame(Parallel(n_jobs=2, batch_size=4)(delayed(one)(r.m, r.alpha, i)
                                                       for r in C.itertuples() for i in range(T.D)))
    R = R.merge(C[["m", "alpha", "post_prob"]], on=["m", "alpha"])
    R["wt"] = R.post_prob / T.D
    plaus = C[C.post_prob >= 0.05]
    res = {}
    for k in CAND:
        diff = R[k] - R["v9"]
        dm = float((R.wt * diff).sum())
        se = float(np.sqrt((R.wt ** 2 * (diff - dm) ** 2).sum()))  # lower bound (draws reuse the same OOF rows)
        worst = min(float(diff[(R.m == p.m) & (R.alpha == p.alpha)].mean()) for p in plaus.itertuples())
        res[k] = (dm, worst)
        say(f"sim (v9 post) {k:10s} expected {(R.wt * R[k]).sum():.2f}  vs v9 {dm:+.2f} (SE {se:.2f})  "
            f"worst plausible cell {worst:+.2f}")
    dm, worst = res["t4_for_h3"]
    say("DECISION (t4_for_h3): " + ("adopt" if dm >= 0.5 and worst >= 0 else "keep v9")
        + f" ({dm:+.2f}, worst cell {worst:+.2f})")
