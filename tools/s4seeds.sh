#!/usr/bin/env bash
# s4 seed ensemble: the s4ff recipe (stage 2 from run4, lr 1e-4 / head 1e-3, 2 epochs, bs 8 x accum 4, ms head,
# 24 layers) with other seeds. Seed averaging of stage-2 fits gave +1.5 on fold 2 before (s2 seeds 42 and 7).
# Usage: bash tools/s4seeds.sh "RUN FOLD SEED" ...   e.g. bash tools/s4seeds.sh "s4ff0s7 0 7" "s4ff1s7 1 7"
#   FOLD 0-3 = out-of-fold run (train.csv minus that fold), -2 = full fit on all of train.csv.
# One run after another through ctl.py (supervisor, resume after a crash). Before each run it checks that at least
# 4 GB of RAM is free (the user's other programs come first) and waits up to 30 min for it, else stops.
# Log: logs/s4seeds.log, ends with "S4SEEDS DONE" or "STOP:".
set -u
cd "$(dirname "$0")/.."
LOG=logs/s4seeds.log
PY=.venv/Scripts/python.exe
MODEL=models/qwen3-4b-base
INIT=checkpoints/run4/final_trainable.pt
say() { echo "$(date '+%Y-%m-%d %H:%M') $*" | tee -a "$LOG"; }
fail() { say "STOP: $*"; exit 1; }
free_gb() { "$PY" -c "import psutil; print(int(psutil.virtual_memory().available / 2**30))" | tr -d ''; }
[[ -f "$MODEL/config.json" ]] || fail "missing $MODEL"
[[ -f "$INIT" ]] || fail "missing $INIT"
[[ $# -ge 1 ]] || fail "no jobs given"
for job in "$@"; do
  set -- $job; run=$1; fold=$2; seed=$3; d=checkpoints/$run
  if [[ -f "$d/DONE_TRAINING" ]]; then say "$run already done, skipping"; continue; fi
  for _ in $(seq 1 60); do (( $(free_gb) >= 4 )) && break; say "waiting: only $(free_gb) GB RAM free"; sleep 30; done
  (( $(free_gb) >= 4 )) || fail "less than 4 GB RAM free for 30 min, not starting $run"
  say "start $run (cv-fold $fold, seed $seed, init $INIT), RAM free $(free_gb) GB"
  "$PY" finetune/ctl.py --run "$run" start --model "$MODEL" --bs 8 --accum 4 --arch ms --layers 24 \
    --lr 1e-4 --head-lr 1e-3 --epochs 2 --eval-n 0 --eval-every 1000000 --ckpt-every 50 \
    --init "$INIT" --seed "$seed" --cv-fold "$fold" >> "$LOG" 2>&1 || fail "ctl start failed for $run"
  low=0
  for _ in $(seq 1 240); do  # 30 s polls, at most 2 h per run (a run takes about 17-25 min)
    [[ -f "$d/DONE_TRAINING" ]] && break  # checked first: a finished run is done even if a PAUSE file also exists
    [[ -f "$d/HALTED" || -f "$d/PAUSE" || -f "$d/STOP" ]] && break
    fm=$("$PY" -c "import psutil; print(int(psutil.virtual_memory().available / 2**20))" | tr -d '')
    if (( fm < 1536 )); then low=$((low + 1)); else low=0; fi
    if (( low >= 2 )); then  # the user's programs need the memory: save and stop, resumable later
      "$PY" finetune/ctl.py --run "$run" pause >> "$LOG" 2>&1
      fail "RAM guard: only ${fm} MB free twice in a row, $run paused (rerun to resume)"
    fi
    sleep 30
  done
  [[ -f "$d/DONE_TRAINING" ]] && { say "$run done: $(tail -1 "$d/train.log")"; continue; }
  [[ -f "$d/HALTED" ]] && fail "$run halted (see $d/supervisor.log)"
  [[ -f "$d/PAUSE" || -f "$d/STOP" ]] && fail "$run was paused or stopped; rerun to resume (finished runs are skipped)"
  "$PY" finetune/ctl.py --run "$run" pause >> "$LOG" 2>&1; fail "$run did not finish within 2 h (pause requested)"
done
say "S4SEEDS DONE"
