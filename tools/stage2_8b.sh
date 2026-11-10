#!/usr/bin/env bash
# 8B stage 2, run one after another: folds 0-3 (OOF for the blend) then the full fit (cv-fold -2) for the notebook.
# Same recipe as the 4B s4ff runs (lr 1e-4 / head 1e-3, 2 epochs, effective batch 32), from k8b_local (8B stage 1).
# bs 4 x accum 8 instead of 8 x 4: the 8B fits the 8 GB card only at bs 4.
# Each run goes through ctl.py (supervisor, resume after a crash, watchdog/governor follow ACTIVE_RUN).
# Log: logs/stage2_8b.log, ends with "STAGE2 8B DONE" or "STOP:".
set -u
cd "$(dirname "$0")/.."
LOG=logs/stage2_8b.log
PY=.venv/Scripts/python.exe
INIT=checkpoints/k8b_local/final_trainable.pt
say() { echo "$(date '+%Y-%m-%d %H:%M') $*" | tee -a "$LOG"; }
fail() { say "STOP: $*"; exit 1; }
[[ -f "$INIT" ]] || fail "missing $INIT"
for job in "k8f0 0" "k8f1 1" "k8f2 2" "k8f3 3" "k8all -2"; do
  set -- $job; run=$1; fold=$2; d=checkpoints/$run
  if [[ -f "$d/DONE_TRAINING" ]]; then say "$run already done, skipping"; continue; fi
  say "start $run (cv-fold $fold)"
  "$PY" finetune/ctl.py --run "$run" start --model models/qwen3-8b-base --bs 4 --accum 8 --arch ms --layers 24 \
    --init "$INIT" --lr 1e-4 --head-lr 1e-3 --epochs 2 --eval-n 0 --eval-every 1000000 --ckpt-every 50 \
    --cv-fold "$fold" >> "$LOG" 2>&1 || fail "ctl start failed for $run"
  for _ in $(seq 1 360); do  # 30 s polls, at most 3 h per run
    [[ -f "$d/DONE_TRAINING" || -f "$d/HALTED" ]] && break
    sleep 30
  done
  [[ -f "$d/HALTED" ]] && fail "$run halted (see $d/supervisor.log)"
  [[ -f "$d/DONE_TRAINING" ]] || fail "$run did not finish within 3 h"
  say "$run done: $(tail -1 "$d/train.log")"
done
say "STAGE2 8B DONE"
