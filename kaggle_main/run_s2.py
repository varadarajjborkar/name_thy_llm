"""Main agent's Kaggle queue: a list of train.py jobs shared by the two T4s (each GPU takes the next job when free).

Inputs: ntl-train-data (train.py), ntl-qwen-weights (init weights <name>__weights.pt), Qwen model, competition.
Outputs per job under /kaggle/working/proj/checkpoints/<job>/: val_logits.npy, val_ids.csv, final_trainable.pt, logs.
Push: cd kaggle_main && kaggle kernels push -p . --accelerator NvidiaTeslaT4
"""
import glob
import os
import shutil
import subprocess
import sys
import time

S2 = ["--lr", "1e-4", "--head-lr", "1e-3", "--epochs", "2", "--eval-n", "0", "--eval-every", "1000000"]
QUEUE = [  # stage 2 from k_host3 (stage 1 on pool + hostlike x3): 4 CV folds for OOF, then the full fit
    ("h3s2all", ["--cv-fold", "-2", "--init", "INIT:kh3final"] + S2),
    ("h3f0", ["--cv-fold", "0", "--init", "INIT:kh3final"] + S2),
    ("h3f1", ["--cv-fold", "1", "--init", "INIT:kh3final"] + S2),
    ("h3f2", ["--cv-fold", "2", "--init", "INIT:kh3final"] + S2),
    ("h3f3", ["--cv-fold", "3", "--init", "INIT:kh3final"] + S2),
]
COMMON = ["--arch", "ms", "--layers", "24", "--compute-dtype", "fp16", "--ckpt-every", "1000000"]

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


shutil.copy(find("train.py", "ntl-train-data"), f"{W}/finetune/train.py")
link(find("train.csv", "name-that-llm"), f"{W}/data/train.csv")
link(os.path.dirname(find("config.json", "4b-base")), f"{W}/models/qwen3-4b-base")
resolve = lambda args: [find(f"{x[5:]}__weights.pt") if x.startswith("INIT:") else x for x in args]

queue, running = list(QUEUE), {}
while queue or running:
    for gpu in (0, 1):
        if gpu not in running and queue:
            run, extra = queue.pop(0)
            args = [sys.executable, f"{W}/finetune/train.py", "--run", run] + COMMON + resolve(extra)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", TRANSFORMERS_VERBOSITY="error")
            os.makedirs(f"{W}/checkpoints/{run}", exist_ok=True)
            log = open(f"{W}/checkpoints/{run}/console.log", "w")
            running[gpu] = (run, subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=W))
            print(f"[{time.strftime('%H:%M')}] started {run} on GPU {gpu}", flush=True)
    time.sleep(30)
    for gpu, (run, p) in list(running.items()):
        if p.poll() is not None:
            R = f"{W}/checkpoints/{run}"
            for d in ("ckpt_latest", "ckpt_prev", "ckpt_tmp"):
                shutil.rmtree(f"{R}/{d}", ignore_errors=True)
            fin = [l for l in open(f"{R}/console.log").read().splitlines() if "FINAL" in l or "Error" in l]
            print(f"[{time.strftime('%H:%M')}] {run} exit {p.returncode}: {fin[-1] if fin else '(no FINAL line)'}", flush=True)
            del running[gpu]
print("ALL DONE", flush=True)
