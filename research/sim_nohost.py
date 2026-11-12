"""Does v9 improve without the only member that saw the hostlike data?
Lineage of the Qwen members in v9 (checkpoints/*/args.json):
  s4 (0.5)   run2 (pool_typo_60k) -> run3 (+pool_cont_30k) -> run4 (+pool_cont2, 92k) -> stage 2 on train.csv
  s2 (0.1)   run2 -> stage 2;   s3 (0.1)  run2 -> run3 -> stage 2
  h3 (0.1)   k_host3 (pool + hostlike x3) -> stage 2         <- the only member trained on hostlike
(The open pool shares a few texts with hostlike: 241 of 60,000 in pool_typo_60k, 0 in pool_cont_30k, 220 of 92,485
in pool_cont2; negligible next to h3's ~20,700 hostlike rows.)
Candidates, fixed in advance:
  v9         0.5 s4 + 0.3 mean(s2, s3, h3) + 0.1 svm + 0.1 lgb + 0.15 atlas     (weights sum 1.15)
  noh3_w0.3  h3 removed, its weight given to s2 and s3:   0.3 mean(s2, s3)      (sum 1.15: the clean comparison)
  noh3_w0.2  h3 removed, its 0.1 weight dropped:          0.2 mean(s2, s3)      (sum 1.05: also ~9% cooler logits
             going into the post, so it mixes "no h3" with a temperature change; reported for reference only)
What the simulator can and cannot see: it scores h3's out-of-fold predictions on train.csv, so whatever h3 learned from
hostlike that helps on train.csv-like text is counted. It cannot know whether the hidden test resembles hostlike more
or less than train.csv; the simulator assumes the test is train.csv-like (as P1-P5 support).
Scored with the v9 post on the calibrated hidden-test simulator (posterior logs/test_sim_v2_cells.csv, cells >= 0.01,
30 draws each, common random numbers), as in research/sim_members.py. Decision rule: adopt only if >= +0.5 vs v9 and
never worse in a plausible cell (posterior >= 0.05).
Run: .venv/Scripts/python.exe research/sim_nohost.py
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

LOG = ROOT / "logs/sim_nohost.log"
FAM = {  # same folders as research/fit_blend_multi.py
    "s2": [ROOT / "research/runs/e005/f0", ROOT / "research/runs/e005/f1", ROOT / "checkpoints/s2f2",
           ROOT / "checkpoints/s2f3"],
    "s3": [ROOT / f"checkpoints/s3f{k}" for k in range(4)],
    "h3": [ROOT / f"checkpoints/h3f{k}" for k in range(4)],
}


def say(*x):
    msg = " ".join(str(v) for v in x)
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def family(dirs):
    parts = []
    for d in dirs:
        vi = pd.read_csv(d / "val_ids.csv").id.astype(str)
        parts.append(pd.DataFrame(T.log_softmax(np.load(d / "val_logits.npy")), index=vi))
    f = pd.concat(parts)
    assert f.index.is_unique and set(f.index) == set(T.ids)
    return f.loc[T.ids].to_numpy()


d, tp = T.d, T.tp
F = {k: family(v) for k, v in FAM.items()}
assert np.allclose((F["s2"] + F["s3"] + F["h3"]) / 3, d["s2mix"]), "families do not rebuild s2mix"
base = 0.5 * d["s4"] + 0.1 * d["svm"] + 0.1 * d["lgb"] + 0.15 * tp["atlas_pool"]
no_h3 = (F["s2"] + F["s3"]) / 2
CAND = {"v9": base + 0.3 * d["s2mix"], "noh3_w0.3": base + 0.3 * no_h3, "noh3_w0.2": base + 0.2 * no_h3}


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
    say("full-structure OOF, families alone (raw argmax): " +
        ", ".join(f"{k} {bacc(y, v.argmax(1)):.4f}" for k, v in F.items()) + f", s4 {bacc(y, d['s4'].argmax(1)):.4f}")
    for k, z in CAND.items():
        say(f"full-structure OOF {k}: raw {bacc(y, z.argmax(1)):.4f}")
    C = pd.read_csv(ROOT / "logs/test_sim_v2_cells.csv")
    C = C[C.post_prob >= 0.01].copy()
    C["post_prob"] /= C.post_prob.sum()
    R = pd.DataFrame(Parallel(n_jobs=2, batch_size=4)(delayed(one)(r.m, r.alpha, i)
                                                       for r in C.itertuples() for i in range(T.D)))
    R = R.merge(C[["m", "alpha", "post_prob"]], on=["m", "alpha"])
    R["wt"] = R.post_prob / T.D
    R.to_parquet(ROOT / "logs/sim_nohost_draws.parquet", index=False)
    plaus = C[C.post_prob >= 0.05]
    res = {}
    for k in CAND:
        diff = R[k] - R["v9"]
        dm = float((R.wt * diff).sum())
        # SE treats draws as independent; they reuse the same OOF rows, so it is a lower bound (issue 105)
        se = float(np.sqrt((R.wt ** 2 * (diff - dm) ** 2).sum()))
        worst = min(float(diff[(R.m == p.m) & (R.alpha == p.alpha)].mean()) for p in plaus.itertuples())
        res[k] = (dm, worst)
        say(f"sim (v9 post) {k:10s} expected {(R.wt * R[k]).sum():.2f}  vs v9 {dm:+.2f} (SE {se:.2f})  "
            f"worst plausible cell {worst:+.2f}")
    dm, worst = res["noh3_w0.3"]
    say("DECISION (on the clean candidate noh3_w0.3): " +
        ("adopt" if dm >= 0.5 and worst >= 0 else "keep v9 with h3") + f" ({dm:+.2f}, worst cell {worst:+.2f})")
