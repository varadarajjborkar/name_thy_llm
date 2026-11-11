#!/usr/bin/env bash
# v7 submission chain (Git Bash). Each step runs only if the previous one passed; every decision is logged.
#   1. wait for the Kaggle PARITY run (kernel version 8) and check ITS output (version-pinned) against the local
#      reference: blend score, ATLAS features, drift, errors, all stages present, parity flag on
#   2. confirm the notebook file is unchanged since the chain started, push it (parity off), find the new version,
#      wait until that run has started and finished, check ITS output: parity off, started after the push,
#      ATLAS in the blend at 0.15, every stage present, no errors, written rows = test rows
#   3. count today's submissions from the date column (UTC day; fail closed) and submit only if fewer than 5;
#      success requires exit code 0, no error text, and one more "v7:" row today in the submissions list than before
#      (newer kaggle CLIs print nothing on success)
# Log: logs/submit_v7.log (the governor watches it; it ends with "SUBMITTED v7" or "STOP:").
set -u
cd "$(dirname "$0")/.."
LOG=logs/submit_v7.log
K=varadarajborkar/ntl-qwen-blend
PARITY_V=8
PY=.venv/Scripts/python.exe
SYSPY="$HOME/AppData/Local/Programs/Python/Python311/python.exe"  # has the kaggle SDK
NB=kernel_qwen/ntl_qwen_blend.py
LOCK=logs/submit_v7.lock.d  # one chain at a time: two chains once submitted the same version twice (2026-11-10)
take_lock() { mkdir "$LOCK" 2>/dev/null && echo $$ > "$LOCK/pid"; }  # mkdir is atomic: only one chain gets it
if ! take_lock; then
  sleep 2; other=$(cat "$LOCK/pid" 2>/dev/null)  # wait out a holder that has made the dir but not written its pid
  if [[ -z "$other" ]] || kill -0 "$other" 2>/dev/null; then echo "another submit chain (pid ${other:-?}) holds $LOCK; not starting"; exit 1; fi
  rm -rf "$LOCK"; take_lock || { echo "lost the race for $LOCK; not starting"; exit 1; }  # stale lock from a dead chain
fi
trap '[[ "$(cat "$LOCK/pid" 2>/dev/null)" == "$$" ]] && rm -rf "$LOCK"' EXIT
say() { echo "$(date '+%Y-%m-%d %H:%M') $*" | tee -a "$LOG"; }
fail() { say "STOP: $*"; exit 1; }
status() { kaggle kernels status "$K" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*' | head -1; }
current_version() {
  "$SYSPY" -c "
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
a = KaggleApi(); a.authenticate()
with a.build_kaggle_client() as k:
    r = ApiGetKernelRequest(); r.user_name, r.kernel_slug = '$K'.split('/')
    print(k.kernels.kernels_api_client.get_kernel(r).to_dict()['metadata']['currentVersionNumber'])
" 2>/dev/null | tail -1
}
wait_done() {  # waits for a terminal status of the latest run; at most 6 h
  local s=""
  for _ in $(seq 1 "$WAIT_LOOPS"); do
    s=$(status)
    case "$s" in *COMPLETE|*ERROR|*CANCEL*) break ;; esac
    sleep 120
  done
  echo "$s"
}
fetch() {  # fetch <version> <dir>: version-pinned output
  rm -rf "${2:?}"; mkdir -p "$2"
  PYTHONIOENCODING=utf-8 kaggle kernels output "$K/$1" -p "$2" -o >/dev/null 2>&1
  [[ -f "$2/run_summary.json" ]]
}

NB_HASH=$(md5sum "$NB" | cut -d' ' -f1)
WAIT_LOOPS=${WAIT_LOOPS:-180}  # 2-minute polls in wait_done (180 = 6 h)
if [[ -n "${RESUME_V:-}" ]]; then  # resume: an already-pushed version is queued; do not push again (that re-queues it)
  [[ -n "${RESUME_PUSH_UTC:-}" ]] || fail "RESUME_V needs RESUME_PUSH_UTC (UTC time of that push)"
  v=$RESUME_V; PUSH_UTC=$RESUME_PUSH_UTC
  say "chain resumed (notebook md5 $NB_HASH): waiting for already-pushed version $v (pushed $PUSH_UTC UTC), up to $((WAIT_LOOPS * 2)) min"
  [[ "$(current_version)" == "$v" ]] || fail "latest kernel version is not $v; something else was pushed"
  grep -q '^PARITY = os.environ.get("NTL_PARITY", "0") == "1"' "$NB" || fail "notebook PARITY default is not off"
  s=$(status)  # the status belongs to the latest version (checked above); a stale terminal status cannot be another run's
  case "$s" in *NEW_SCRIPT|*QUEUED|*RUNNING|*COMPLETE) ;; *) fail "version $v is not queued, running or complete (status $s)" ;; esac
elif [[ "${SKIP_PARITY:-0}" == "1" ]]; then  # user decision 2026-11-09: save Kaggle GPU quota, rely on the local tests
  say "chain started (notebook md5 $NB_HASH): Kaggle parity skipped (local parity tests passed); waiting for version $PARITY_V to stop"
  s=$(wait_done); say "version $PARITY_V status: $s"
else
say "chain started (notebook md5 $NB_HASH): waiting for the parity run (version $PARITY_V)"
[[ "$(current_version)" == "$PARITY_V" ]] || fail "latest kernel version is not $PARITY_V; something else was pushed"
s=$(wait_done); say "parity run status: $s"
[[ "$s" == *COMPLETE ]] || fail "parity run did not complete"
[[ "$(current_version)" == "$PARITY_V" ]] || fail "a newer version appeared while waiting"
fetch "$PARITY_V" logs/v7_parity_out || fail "could not download the version $PARITY_V output"
$PY - logs/v7_parity_out <<'EOF' >> "$LOG" 2>&1 || fail "parity checks failed (see lines above)"
import json, sys
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
parity_on = s["stages"][0].get("parity") is True
print("parity flag:", parity_on, "| parity blend:", blend, "| local reference raw 0.8222 balanced 0.8800 | drift:",
      s.get("atlas_drift"), "| errors:", errors)
k = pd.read_parquet(f"{out}/atlas_parity_features.parquet")
h = pd.read_parquet("data/processed/atlas_host_fp16.parquet"); h["id"] = h.id.astype(str)
d = np.abs(k[COLS].to_numpy() - h.set_index("id").loc[k.id.astype(str), COLS].to_numpy())
print(f"ATLAS features Kaggle vs local: median {np.median(d):.5f} p99 {np.quantile(d, .99):.4f} max {d.max():.4f}")
ok = (parity_on and not missing and not errors and abs(blend.get("raw", 0) - 0.8222) < 0.02
      and np.median(d) < 0.01 and np.quantile(d, .99) < 0.2)
print("PARITY CHECK", "PASSED" if ok else f"FAILED (missing stages {missing})")
sys.exit(0 if ok else 1)
EOF
say "parity checks passed"
fi

if [[ -z "${RESUME_V:-}" ]]; then
[[ "$(md5sum "$NB" | cut -d' ' -f1)" == "$NB_HASH" ]] || fail "notebook changed since the chain started"
grep -q '^PARITY = os.environ.get("NTL_PARITY", "0") == "1"' "$NB" || fail "notebook PARITY default is not off"
PUSH_UTC=$(date -u '+%Y-%m-%d %H:%M:%S')
out=$(kaggle kernels push -p kernel_qwen 2>&1 | grep -v Warning)
say "push: $out"
sleep 30
v=$(current_version)
[[ -n "$v" && "$v" -gt "$PARITY_V" ]] || fail "no new kernel version after the push (latest: '$v')"
say "real notebook is version $v; waiting for its run to start"
started=0
for _ in $(seq 1 40); do  # the status shows the previous run (COMPLETE or CANCEL) until the new one starts
  s=$(status); case "$s" in *QUEUED|*RUNNING) started=1; break ;; esac; sleep 30
done
[[ $started -eq 1 ]] || fail "version $v did not start within 20 minutes (last status $s)"
fi
s=$(wait_done); say "real run (version $v) status: $s"
[[ "$s" == *COMPLETE ]] || fail "real run did not complete"
fetch "$v" logs/v7_real_out || fail "could not download the version $v output"
$PY - logs/v7_real_out "$PUSH_UTC" <<'EOF' >> "$LOG" 2>&1 || fail "real run checks failed (see lines above)"
import json, sys
s = json.load(open(f"{sys.argv[1]}/run_summary.json"))
st = [x["stage"] for x in s["stages"]]
w = s.get("weights", {})
first = s["stages"][0]
written = next((x for x in s["stages"] if x["stage"] == "written"), {})
print("real run started", s["started"], "(push", sys.argv[2], ") | stages:", st, "| weights:", w)
ok = (s["started"] >= sys.argv[2] and first.get("parity") is False and "atlas" in st and "written" in st
      and w.get("atlas") == 0.15 and w.get("qwen:k8all") == 0.2 and "qwen k8all" in st
      and written.get("rows") == first.get("test")
      and not any("error" in k for k in s))
print("REAL RUN CHECK", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
EOF
say "real run checks passed"

today=$(date -u +%Y-%m-%d)
table=$(kaggle competitions submissions -c name-that-llm --csv 2>&1)
echo "$table" | grep -q '^ref,fileName,date' || fail "could not read the submissions list"
n=$(echo "$table" | awk -F, -v d="$today" '$1 ~ /^[0-9]+$/ && substr($3, 1, 10) == d {c++} END {print c + 0}')
say "submissions today (UTC $today) before submitting: $n"
[[ "$n" -lt 5 ]] || fail "daily limit reached; submit after 05:30 IST"
V7ROWS='$1 ~ /^[0-9]+$/ && substr($3, 1, 10) == d && /"v8: v7 \+ Qwen3-8B/ {c++} END {print c + 0}'
n1=$(echo "$table" | awk -F, -v d="$today" "$V7ROWS")  # v7 rows already there today: success needs one more
res=$(kaggle competitions submit name-that-llm -k "$K" -v "$v" -f submission.parquet \
  -m "v8: v7 + Qwen3-8B stage-2 member at 0.2 (OOF +0.88, all 4 folds up; fp16 check 99.5% agreement)" 2>&1)
rc=$?
say "submit (rc $rc): $(echo "$res" | grep -v Warning)"
# newer kaggle CLIs print nothing on success (2026-11-10), so the submissions list is the proof, not the message
[[ $rc -eq 0 ]] && ! echo "$res" | grep -qiE "error|not allowed" || fail "submit was not accepted"
sleep 20
n2=$(kaggle competitions submissions -c name-that-llm --csv 2>&1 | awk -F, -v d="$today" "$V7ROWS")
[[ "$n2" -gt "$n1" ]] || fail "no v8 row for today in the submissions list (check by hand before resubmitting)"
say "SUBMITTED v8 (kernel version $v)"
