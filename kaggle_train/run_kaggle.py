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

MODE = "real8b"  # smoke | real | smoke8b | real8b
# 8B smoke (2026-11-09): 24 layers about 15 s/step at batch 32 on T4, 32 layers about 19 s/step, same accuracy
# after 40 steps (0.204). Real runs keep 24 layers: 60k rows is about 8 h, inside the 12 h session limit.
Q8 = ["--model", "/kaggle/working/proj/models/qwen3-8b-base"]
# Smoke test result (2026-11-09): fp16 and bf16 reach the same accuracy (0.152 vs 0.151 after 40 steps),
# fp16 is ~4x faster on T4 (no native bf16). Real runs use fp16.
# Each job: (run name, training parquet built below, extra args)
JOBS = {
    "smoke": [
        ("k_fp16", "pool_typo_60k.parquet", ["--compute-dtype", "fp16", "--n-train", "640", "--bs", "8", "--accum", "2"]),
        ("k_bf16", "pool_typo_60k.parquet", ["--compute-dtype", "bf16", "--n-train", "640", "--bs", "8", "--accum", "2"]),
    ],
    "smoke8b": [  # Qwen3-8B speed and memory on T4: 24 vs 32 kept layers
        ("k8_l24", "pool_typo_60k.parquet", Q8 + ["--compute-dtype", "fp16", "--n-train", "1280", "--layers", "24"]),
        ("k8_l32", "pool_typo_60k.parquet", Q8 + ["--compute-dtype", "fp16", "--n-train", "1280", "--layers", "32",
                                                  "--depths", "4,8,12,16,20,24,28,32"]),
    ],
    "real8b": [  # Qwen3-8B stage 1: same data as run2 (direct 4B vs 8B comparison), and pool + hostlike
        ("k8a", "pool_typo_60k.parquet", Q8 + ["--compute-dtype", "fp16", "--layers", "24", "--eval-every", "250"]),
        ("k8b", "pool_host1.parquet", Q8 + ["--compute-dtype", "fp16", "--layers", "24", "--eval-every", "250"]),
    ],
    "real": [  # data experiments against local run2 (pool only, same recipe and seed)
        ("k_host1", "pool_host1.parquet", ["--compute-dtype", "fp16", "--eval-every", "250"]),
        ("k_host3", "pool_host3.parquet", ["--compute-dtype", "fp16", "--eval-every", "250"]),
    ],
}
COMMON = ["--arch", "ms", "--layers", "24", "--lr", "1e-4", "--head-lr", "1e-3", "--warmup", "0.1",
          "--lockbox-fold", "4", "--eval-n", "900", "--ckpt-every", "100",
          "--max-hours", "10.75"]  # Kaggle stops a session at 12 h: save a resumable checkpoint well before

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
if glob.glob("/kaggle/input/**/8b-base/**/config.json", recursive=True):
    link(os.path.dirname(find("config.json", "8b-base")), f"{W}/models/qwen3-8b-base")

# pool + host-faithful extra rows (hostlike has no host prompts; checked in prep_hostlike.py)
import pandas as pd  # noqa: E402
pool = pd.read_parquet(f"{W}/data/processed/pool_typo_60k.parquet", columns=["text", "model"])
host = pd.read_parquet(f"{W}/data/processed/hostlike.parquet", columns=["text", "model"])
for k in (1, 3):
    mix = pd.concat([pool] + [host] * k, ignore_index=True).sample(frac=1, random_state=0)
    mix.to_parquet(f"{W}/data/processed/pool_host{k}.parquet", index=False)
    print(f"pool_host{k}: {len(mix):,} rows", flush=True)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout, flush=True)

procs = []
for gpu, (run, train_file, extra) in enumerate(JOBS[MODE]):
    args = [sys.executable, f"{W}/finetune/train.py", "--run", run,
            "--train", f"{W}/data/processed/{train_file}"] + COMMON + extra
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
    if p.returncode == 0:  # finished: optimizer state is not needed afterwards; keep outputs small
        for d in ("ckpt_latest", "ckpt_prev", "ckpt_tmp"):
            shutil.rmtree(f"{W}/checkpoints/{run}/{d}", ignore_errors=True)
    else:  # paused at the time limit, aborted or crashed: keep the checkpoints so the run can be resumed
        print(f"{run}: kept ckpt_latest/ckpt_prev for a resume", flush=True)
    tail = open(f"{W}/checkpoints/{run}/console.log").read().splitlines()[-15:]
    print("\n".join(tail), flush=True)
