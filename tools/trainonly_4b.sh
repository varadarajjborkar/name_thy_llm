#!/usr/bin/env bash
# Qwen3-4B fine-tuned on train.csv ONLY (no stage 1, no external data), one run after another: folds 0-3 (OOF for the
# comparison) then the full fit (cv-fold -2) for a possible notebook member.
# Same recipe as the 4B s4ff runs (lr 1e-4 / head 1e-3, 2 epochs, bs 8 x accum 4, ms head, 24 layers) but no --init:
# it starts from the plain base model, so the only difference to s4ff is the missing stage 1. Answers "is train.csv enough?".
# Each run goes through ctl.py (supervisor, resume after a crash, watchdog/governor follow ACTIVE_RUN).
# Log: logs/trainonly_4b.log, ends with "TRAINONLY 4B DONE" or "STOP:".
set -u
cd "$(dirname "$0")/.."
LOG=logs/trainonly_4b.log
PY=.venv/Scripts/python.exe
MODEL=models/qwen3-4b-base
say() { echo "$(date '+%Y-%m-%d %H:%M') $*" | tee -a "$LOG"; }
fail() { say "STOP: $*"; exit 1; }
[[ -f "$MODEL/config.json" ]] || fail "missing $MODEL"
for job in "t4f0 0" "t4f1 1" "t4f2 2" "t4f3 3" "t4all -2"; do
  set -- $job; run=$1; fold=$2; d=checkpoints/$run
  if [[ -f "$d/DONE_TRAINING" ]]; then say "$run already done, skipping"; continue; fi
  say "start $run (cv-fold $fold)"
  "$PY" finetune/ctl.py --run "$run" start --model "$MODEL" --bs 8 --accum 4 --arch ms --layers 24 \
    --lr 1e-4 --head-lr 1e-3 --epochs 2 --eval-n 0 --eval-every 1000000 --ckpt-every 50 \
    --cv-fold "$fold" >> "$LOG" 2>&1 || fail "ctl start failed for $run"
  for _ in $(seq 1 240); do  # 30 s polls, at most 2 h per run (a run takes about 17 min)
    [[ -f "$d/DONE_TRAINING" ]] && break  # checked first: a finished run is done even if a PAUSE file also exists
    [[ -f "$d/HALTED" || -f "$d/PAUSE" || -f "$d/STOP" ]] && break
    sleep 30
  done
  [[ -f "$d/DONE_TRAINING" ]] && { say "$run done: $(tail -1 "$d/train.log")"; continue; }
  [[ -f "$d/HALTED" ]] && fail "$run halted (see $d/supervisor.log)"
  [[ -f "$d/PAUSE" || -f "$d/STOP" ]] && fail "$run was paused or stopped; rerun this script to resume (finished runs are skipped)"
  "$PY" finetune/ctl.py --run "$run" pause >> "$LOG" 2>&1; fail "$run did not finish within 2 h (pause requested)"
done
say "TRAINONLY 4B DONE"
