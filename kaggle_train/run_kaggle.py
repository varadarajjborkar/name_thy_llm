"""Run finetune/train.py on Kaggle's 2x T4, one job per GPU, in parallel.

Inputs: dataset varadarajborkar/ntl-train-data (train.py + training parquets), model qwen-lm/qwen-3 4b-base,
competition name-that-llm (train.csv). Internet on (bitsandbytes is not preinstalled).
Recreates the local folder layout under /kaggle/working/proj so train.py runs unchanged.
Outputs: /kaggle/working/proj/checkpoints/<run>/ (logs, metrics, eval logits, final_trainable.pt).
"""
import glob
import os
import shutil
import subprocess
import sys
import time

MODE = "smoke"  # smoke | real
JOBS = {
    "smoke": [
        ("k_fp16", ["--compute-dtype", "fp16", "--n-train", "640", "--bs", "8", "--accum", "2"]),
        ("k_bf16", ["--compute-dtype", "bf16", "--n-train", "640", "--bs", "8", "--accum", "2"]),
    ],
    "real": [],  # filled in after the smoke test
}
COMMON = ["--arch", "ms", "--layers", "24", "--lr", "1e-4", "--head-lr", "1e-3", "--warmup", "0.1",
          "--lockbox-fold", "4", "--eval-n", "900", "--ckpt-every", "100"]

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=True)
W = "/kaggle/working/proj"
for d in ("finetune", "data/processed", "models", "checkpoints"):
    os.makedirs(f"{W}/{d}", exist_ok=True)


def find(pattern, must=""):
    hits = [h for h in sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)) if must in h]
    assert hits, f"{pattern} ({must}) not found"
    return hits[0]


def link(src, dst):
    if not os.path.exists(dst):
        os.symlink(src, dst)


shutil.copy(find("train.py"), f"{W}/finetune/train.py")
link(find("train.csv", "name-that-llm"), f"{W}/data/train.csv")
for f in ("pool_typo_60k.parquet", "hostlike.parquet"):
    link(find(f), f"{W}/data/processed/{f}")
link(os.path.dirname(find("config.json", "4b-base")), f"{W}/models/qwen3-4b-base")
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout, flush=True)

procs = []
for gpu, (run, extra) in enumerate(JOBS[MODE]):
    args = [sys.executable, f"{W}/finetune/train.py", "--run", run,
            "--train", f"{W}/data/processed/pool_typo_60k.parquet"] + COMMON + extra
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", TRANSFORMERS_VERBOSITY="error")
    os.makedirs(f"{W}/checkpoints/{run}", exist_ok=True)
    log = open(f"{W}/checkpoints/{run}/console.log", "w")
    procs.append((run, subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=W)))
    print(f"started {run} on GPU {gpu}: {' '.join(args[2:])}", flush=True)

while any(p.poll() is None for _, p in procs):
    time.sleep(60)
    for run, _ in procs:
        lines = open(f"{W}/checkpoints/{run}/train.log").read().splitlines() if os.path.exists(f"{W}/checkpoints/{run}/train.log") else []
        print(f"[{time.strftime('%H:%M')}] {run}: {lines[-1] if lines else '(starting)'}", flush=True)

for run, p in procs:
    print(f"{run} exit code {p.returncode}", flush=True)
    for d in ("ckpt_latest", "ckpt_prev", "ckpt_tmp"):  # optimizer state is not needed afterwards; keep outputs small
        shutil.rmtree(f"{W}/checkpoints/{run}/{d}", ignore_errors=True)
    tail = open(f"{W}/checkpoints/{run}/console.log").read().splitlines()[-15:]
    print("\n".join(tail), flush=True)
