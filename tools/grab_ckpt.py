"""Keep LoRA + head weights of a running training run at chosen steps, for checkpoint ensembles.

The trainer keeps only rolling checkpoints (latest + previous, every --ckpt-every steps). This copies the trainable
tensors out of them when the wanted step appears, without touching the run.
Run: .venv/Scripts/python.exe tools/grab_ckpt.py run2 1300 1500 1700
Output: checkpoints/<run>/keep/step<N>.pt
"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
run, wanted = sys.argv[1], {int(s) for s in sys.argv[2:]}
R = ROOT / "checkpoints" / run
(R / "keep").mkdir(exist_ok=True)
while wanted:
    for d in ("ckpt_latest", "ckpt_prev"):
        ok = R / d / "OK"
        try:
            step = int(ok.read_text())
        except (OSError, ValueError):
            continue
        if step in wanted:
            st = torch.load(R / d / "state.pt", map_location="cpu", weights_only=False)
            if st["step"] != step:  # rotated while reading
                continue
            tmp = R / "keep" / f"step{step}.tmp"
            torch.save(st["trainable"], tmp)
            tmp.replace(R / "keep" / f"step{step}.pt")
            wanted.discard(step)
            print(f"kept step {step}", flush=True)
    if (R / "DONE_TRAINING").exists() and wanted:
        print(f"run finished; never saw steps {sorted(wanted)}", flush=True)
        break
    time.sleep(60)
