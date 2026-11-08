"""Experiment log: every training decision, the evidence behind it, and the result. Committed with the code.

  python tools/explog.py add "<decision>" "<evidence>" [result]
  python tools/explog.py list

Rules this log exists to enforce:
  * decisions use validation numbers (never training loss alone, never the lockbox fold 4)
  * one change at a time, so the result is attributable
  * leaderboard gains under about 0.01 are noise (identical code varies by up to 0.009)
"""
import json
import sys
import time
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "experiments.jsonl"

if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["add"] and len(a) >= 3:
        rec = {"time": time.strftime("%Y-%m-%d %H:%M"), "decision": a[1], "evidence": a[2],
               "result": a[3] if len(a) > 3 else ""}
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print("logged")
    elif a[:1] == ["list"]:
        for line in LOG.read_text(encoding="utf-8").splitlines() if LOG.exists() else []:
            r = json.loads(line)
            print(f"{r['time']}  {r['decision']}\n    evidence: {r['evidence']}\n    result: {r['result'] or '(pending)'}")
    else:
        print(__doc__)
