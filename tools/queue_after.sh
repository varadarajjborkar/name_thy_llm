#!/usr/bin/env bash
# Wait until a run is DONE_TRAINING (or its chain stopped), then start tools/s4seeds.sh with the given jobs.
# Usage: bash tools/queue_after.sh RUN_TO_WAIT_FOR "JOB" ...   Keeps one trainer on the GPU at a time.
set -u
cd "$(dirname "$0")/.."
wait_for=$1; shift
start_line=$(wc -l < logs/s4seeds.log 2>/dev/null || echo 0)  # only STOP lines written after this start count
new_stop() { tail -n +"$((start_line + 1))" logs/s4seeds.log 2>/dev/null | grep -q "STOP"; }
until [[ -f "checkpoints/$wait_for/DONE_TRAINING" ]] || new_stop; do sleep 30; done
if [[ ! -f "checkpoints/$wait_for/DONE_TRAINING" ]] && new_stop; then echo "$(date '+%Y-%m-%d %H:%M') queue_after: chain stopped, not starting" >> logs/s4seeds.log; exit 1; fi
sleep 60  # let the finished trainer release the GPU and RAM
exec bash tools/s4seeds.sh "$@"
