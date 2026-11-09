#!/usr/bin/env bash
# v7 submission chain (Git Bash). Each step runs only if the previous one passed; every decision is logged.
#   1. wait for the Kaggle PARITY run (kernel version 8) to finish
#   2. check it against the local reference: blend score, ATLAS features, drift, errors, all stages present
#   3. push the real notebook (PARITY off), wait for its run, check its log
#   4. count today's submissions (UTC day) and submit only if fewer than 5
# Log: logs/submit_v7.log (the governor watches it).
set -u
cd "$(dirname "$0")/.."
LOG=logs/submit_v7.log
OUT=logs/v7_parity_out
OUT2=logs/v7_real_out
K=varadarajborkar/ntl-qwen-blend
PY=.venv/Scripts/python.exe
say() { echo "$(date '+%Y-%m-%d %H:%M') $*" | tee -a "$LOG"; }
fail() { say "STOP: $*"; exit 1; }

wait_kernel() {  # prints the final status
  local s=""
  for _ in $(seq 1 150); do
    s=$(kaggle kernels status "$K" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*' | head -1)
    case "$s" in *COMPLETE|*ERROR|*CANCEL*) break ;; esac
    sleep 120
  done
  echo "$s"
}

say "chain started: waiting for the parity run (version 8)"
s=$(wait_kernel); say "parity run status: $s"
[[ "$s" == *COMPLETE ]] || fail "parity run did not complete"
rm -rf "$OUT"; mkdir -p "$OUT"
PYTHONIOENCODING=utf-8 kaggle kernels output "$K" -p "$OUT" -o >/dev/null 2>&1
$PY - "$OUT" <<'EOF' >> "$LOG" 2>&1 || fail "parity checks failed (see lines above)"
import glob, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "features")
from atlas_feats import COLS
out = sys.argv[1]
s = json.load(open(f"{out}/run_summary.json"))
stages = [x["stage"] for x in s["stages"]]
print("kaggle stages:", [(x["stage"], x["t"]) for x in s["stages"]])
need = ["data", "qwen s4all", "qwen s2all", "qwen s3all", "qwen h3s2all", "lightgbm", "atlas", "prompt centering", "written"]
missing = [n for n in need if n not in stages]
errors = {k: v for k, v in s.items() if "error" in k}
blend = s.get("parity", {}).get("blend", {})
print("parity blend:", blend, "| local reference raw 0.8222 balanced 0.8800 | drift:", s.get("atlas_drift"), "| errors:", errors)
k = pd.read_parquet(f"{out}/atlas_parity_features.parquet")
h = pd.read_parquet("data/processed/atlas_host_fp16.parquet"); h["id"] = h.id.astype(str)
d = np.abs(k[COLS].to_numpy() - h.set_index("id").loc[k.id.astype(str), COLS].to_numpy())
print(f"ATLAS features Kaggle vs local: median {np.median(d):.5f} p99 {np.quantile(d, .99):.4f} max {d.max():.4f}")
ok = (not missing and not errors and abs(blend.get("raw", 0) - 0.8222) < 0.02 and np.median(d) < 0.01
      and np.quantile(d, .99) < 0.2)
print("PARITY CHECK", "PASSED" if ok else f"FAILED (missing stages {missing})")
sys.exit(0 if ok else 1)
EOF
say "parity checks passed; pushing the real notebook"
v=$(kaggle kernels push -p kernel_qwen 2>&1 | grep -o 'Kernel version [0-9]*' | grep -o '[0-9]*$')
[[ -n "$v" ]] || fail "push of the real notebook failed"
say "real notebook pushed as version $v"
sleep 60
s=$(wait_kernel); say "real run (version $v) status: $s"
[[ "$s" == *COMPLETE ]] || fail "real run did not complete"
rm -rf "$OUT2"; mkdir -p "$OUT2"
PYTHONIOENCODING=utf-8 kaggle kernels output "$K" -p "$OUT2" -o >/dev/null 2>&1
$PY -c "
import json, sys
s = json.load(open('$OUT2/run_summary.json'))
st = [x['stage'] for x in s['stages']]
w = s.get('weights', {})
print('real run stages:', st, '| weights:', w)
ok = 'written' in st and 'atlas' in st and w.get('atlas') == 0.15 and not any('error' in k for k in s)
sys.exit(0 if ok else 1)
" >> "$LOG" 2>&1 || fail "real run log check failed"
n=$(kaggle competitions submissions -c name-that-llm 2>&1 | grep -c "$(date -u +%Y-%m-%d)")
say "submissions today (UTC $(date -u +%Y-%m-%d)) before submitting: $n"
[[ "$n" -lt 5 ]] || fail "daily limit reached; submit after 05:30 IST"
kaggle competitions submit name-that-llm -k "$K" -v "$v" -f submission.parquet \
  -m "v7: v6 + ATLAS base-LM likelihood member at 0.15 (OOF +1.08 seed-averaged, every fold up); Kaggle parity checked" \
  2>&1 | grep -v Warning | tee -a "$LOG"
say "SUBMITTED v7 (kernel version $v)"
