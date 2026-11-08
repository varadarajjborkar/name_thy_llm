"""Block until the active run logs a new EVAL / FINAL / FATAL line, or stops. Used to wake the agent between evals.

  python tools/wait_eval.py [run] [max_minutes]
Prints the new lines and why it returned.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
run = sys.argv[1] if len(sys.argv) > 1 else (ROOT / "checkpoints" / "ACTIVE_RUN").read_text().strip()
max_min = float(sys.argv[2]) if len(sys.argv) > 2 else 90
RUN = ROOT / "checkpoints" / run
log = RUN / "train.log"
KEYS = ("EVAL", "FINAL", "FATAL", "too many non-finite")


def lines():
    return log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []


start_n, t0 = len(lines()), time.time()
while time.time() - t0 < max_min * 60:
    new = lines()[start_n:]
    hits = [l for l in new if any(k in l for k in KEYS)]
    if hits:
        print(f"[{run}] new event:")
        print("\n".join(new[-6:]))
        sys.exit(0)
    for f, why in (("DONE_TRAINING", "finished"), ("HALTED", "halted"), ("STOP", "stopped by user")):
        if (RUN / f).exists():
            print(f"[{run}] {why}")
            print("\n".join(lines()[-6:]))
            sys.exit(0)
    time.sleep(20)
print(f"[{run}] no eval within {max_min} min; last lines:")
print("\n".join(lines()[-4:]))
