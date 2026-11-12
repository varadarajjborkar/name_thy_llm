"""Fourth paired LB probe for the hidden-test simulator (research/test_sim.py): v7 vs v5.
  v5 = 3-family Qwen blend (q3) + centering 0.5 + FULL sinkhorn            LB 0.572
  v7 = v7 members + centering 0.5 + partial sinkhorn 0.45                 LB 0.571   -> P4 = -0.1
Unlike P1-P3 this pair also changes members (s4 and ATLAS added), so it also measures how much the member gains
carry over to the hidden test. Same draws as test_sim.py (common random numbers: seed 10_000 + i), one worker
(low RAM while the GPU trains). Reads logs/test_sim_draws.parquet, writes logs/test_sim_p4.parquet, then refits the
posterior with P1-P4 and re-ranks the v7 post settings.
Run: .venv/Scripts/python.exe research/test_sim_p4.py
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import balanced_accuracy_score as bacc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import test_sim as T  # noqa: E402

try:
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass
LOG = ROOT / "logs/test_sim_p4.log"
LOG.write_text("", encoding="utf-8")


def say(*x):
    msg = " ".join(str(v) for v in x)
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


R = pd.read_parquet(ROOT / "logs/test_sim_draws.parquet")
out = ROOT / "logs/test_sim_p4.parquet"
rows = []
for n, r in enumerate(R.to_dict("records")):
    keep = T.draw(r["m"], r["alpha"], np.random.default_rng(10_000 + int(r["i"])))
    assert len(keep) == r["rows"], "draw mismatch with test_sim.py"
    yy, g = T.y[keep], T.find_groups(T.text[keep], 18)
    v5 = T.post(T.PIPE["q3"][keep], g, 0.5, 1.0).argmax(1)
    v7 = T.post(T.PIPE["v7"][keep], g, 0.5, 0.45).argmax(1)
    assert abs(100 * bacc(yy, v7) - r[T.SHIPPED]) < 1e-6, "v7 shipped score differs from test_sim.py"
    rows.append({"m": r["m"], "alpha": r["alpha"], "i": r["i"], "P4_v7_v5": 100 * (bacc(yy, v7) - bacc(yy, v5)),
                 "f_P4_v7_v5": float(np.mean((v7 == yy) != (v5 == yy)))})
    if n % 100 == 0:
        say(f"{n}/{len(R)}")
P4 = pd.DataFrame(rows)
P4.to_parquet(out, index=False)
R = R.merge(P4, on=["m", "alpha", "i"])
LB = dict(T.LB, P4_v7_v5=-0.1)
cells = []
for (m, a), g in R.groupby(["m", "alpha"]):
    c = {"m": m, "alpha": a, "per_prompt": g.per_prompt.mean()}
    for name, probes in (("chi2_p123", list(T.LB)), ("chi2_p1234", list(LB))):
        chi = 0.0
        for k in probes:
            sig2 = (100 * np.sqrt(g[f"f_{k}"].mean() / T.N_PUBLIC)) ** 2 + g[k].var() / len(g)
            chi += (g[k].mean() - LB[k]) ** 2 / sig2
        c[name] = chi
    c["P4_v7_v5"] = g.P4_v7_v5.mean()
    cells.append(c)
C = pd.DataFrame(cells)
for name in ("chi2_p123", "chi2_p1234"):
    p = np.exp(-(C[name] - C[name].min()) / 2)
    C["prob_" + name[5:]] = p / p.sum()
say("cells sorted by chi2 with all four probes (P4 = v7 - v5, LB -0.1):")
say(C.sort_values("chi2_p1234").round(3).head(12).to_string(index=False))
say(f"\nposterior mean P4 (sim) with P1-P3: {(C.P4_v7_v5 * C.prob_p123).sum():+.2f}; LB -0.1. "
    "The gap is the part of v7's member gains that does not reach the hidden test.")
W = R.merge(C[["m", "alpha", "prob_p1234"]], on=["m", "alpha"])
W["wt"] = W.prob_p1234 / T.D
keys = [c for c in R.columns if c.startswith("w") and "_a" in c]
res = []
plaus = C[C.prob_p1234 >= 0.05][["m", "alpha"]]
for k in keys:
    diff = W[k] - W[T.SHIPPED]
    dm = float((W.wt * diff).sum())
    se = float(np.sqrt((W.wt ** 2 * (diff - dm) ** 2).sum()))
    worst = min(float(diff[(W.m == r.m) & (W.alpha == r.alpha)].mean()) for r in plaus.itertuples())
    res.append({"setting": k, "expected": float((W.wt * W[k]).sum()), "vs_shipped": dm, "se": se,
                "worst_plausible_cell": worst})
S = pd.DataFrame(res).sort_values("expected", ascending=False)
S.to_csv(ROOT / "logs/test_sim_post_p4.csv", index=False)
say(f"\nv7 post settings under the P1-P4 posterior (shipped {T.SHIPPED}):")
say(S.head(12).round(3).to_string(index=False))
