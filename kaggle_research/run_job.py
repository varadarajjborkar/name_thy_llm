"""Autoresearch job runner: up to two train_exp.py jobs in parallel on Kaggle's 2x T4, then a metrics summary.

Edit EXP and JOBS, then push:  cd kaggle_research && kaggle kernels push -p . --accelerator NvidiaTeslaT4
Inputs: dataset ntl-research-code (train_exp.py, assign.py), ntl-qwen-weights (init weights), ntl-train-data
(hostlike.parquet, pool parquet), model qwen-lm/qwen-3 4b-base, competition name-that-llm.
Output: /kaggle/working/summary.json  {job: {raw, sinkhorn, grouped, n_val, minutes, exit}} plus each job's logs.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

EXP = "e012-pseudo0.4"  # experiment id, also printed in summary.json
JOBS = [  # (job name, extra args for train_exp.py); job i runs on GPU i
    ("f0", ["--cv-fold","0","--init","INIT:run2final","--lr","1e-4","--head-lr","1e-3","--epochs","2","--pseudo-frac","0.4","--pseudo","CODE:pseudo_e005e010_f0.npy"]),
    ("f1", ["--cv-fold","1","--init","INIT:run2final","--lr","1e-4","--head-lr","1e-3","--epochs","2","--pseudo-frac","0.4","--pseudo","CODE:pseudo_e005e010_f1.npy"]),
]
COMMON = ["--arch", "ms", "--layers", "24", "--compute-dtype", "fp16", "--eval-n", "0", "--ckpt-every", "1000000",
          "--eval-every", "1000000"]

t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=True)  # internet is on here
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


shutil.copy(find("train_exp.py"), f"{W}/finetune/train.py")
sys.path.insert(0, os.path.dirname(find("assign.py")))
from assign import assign_groups, find_groups, sinkhorn  # noqa: E402

link(find("train.csv", "name-that-llm"), f"{W}/data/train.csv")
for f in glob.glob("/kaggle/input/**/*.parquet", recursive=True):
    link(f, f"{W}/data/processed/{os.path.basename(f)}")
link(os.path.dirname(find("config.json", "4b-base")), f"{W}/models/qwen3-4b-base")


def resolve(args):  # INIT:<name> -> path of <name>__weights.pt in ntl-qwen-weights; CODE:<file> -> file in ntl-research-code
    return [find(f"{x[5:]}__weights.pt") if x.startswith("INIT:") else find(x[5:]) if x.startswith("CODE:") else x
            for x in args]


procs = []
for gpu, (run, extra) in enumerate(JOBS):
    args = [sys.executable, f"{W}/finetune/train.py", "--run", run] + COMMON + resolve(extra)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", TRANSFORMERS_VERBOSITY="error")
    os.makedirs(f"{W}/checkpoints/{run}", exist_ok=True)
    log = open(f"{W}/checkpoints/{run}/console.log", "w")
    procs.append((run, subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=W), time.time()))
    print(f"started {run} on GPU {gpu}: {' '.join(args[2:])}", flush=True)

while any(p.poll() is None for _, p, _ in procs):
    time.sleep(60)
    for run, _, _ in procs:
        f = f"{W}/checkpoints/{run}/train.log"
        lines = open(f).read().splitlines() if os.path.exists(f) else []
        print(f"[{time.strftime('%H:%M')}] {run}: {lines[-1] if lines else '(starting)'}", flush=True)

from sklearn.metrics import balanced_accuracy_score as bacc  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

summary = {"exp": EXP, "jobs": {}, "total_minutes": None}
for run, p, ts in procs:
    R = f"{W}/checkpoints/{run}"
    for d in ("ckpt_latest", "ckpt_prev", "ckpt_tmp"):
        shutil.rmtree(f"{R}/{d}", ignore_errors=True)
    res = {"exit": p.returncode, "minutes": round((time.time() - ts) / 60, 1)}
    if os.path.exists(f"{R}/val_logits.npy"):
        lg = np.load(f"{R}/val_logits.npy")
        ids = pd.read_csv(f"{R}/val_ids.csv")
        classes = json.load(open(f"{R}/classes.json"))
        y = ids.model.map({c: i for i, c in enumerate(classes)}).to_numpy()
        texts = pd.read_csv(f"{W}/data/train.csv").set_index("id").loc[ids.id, "text"].fillna("").values
        res.update(n_val=len(y), raw=bacc(y, lg.argmax(1)), sinkhorn=bacc(y, sinkhorn(lg).argmax(1)),
                   grouped_oracle=bacc(y, assign_groups(lg, ids.prompt_id.to_numpy())),
                   grouped=bacc(y, assign_groups(lg, find_groups(texts))))
    else:
        res["tail"] = open(f"{R}/console.log").read().splitlines()[-25:]
    summary["jobs"][run] = res
    print(run, json.dumps({k: v for k, v in res.items() if k != "tail"}), flush=True)
    if "tail" in res:
        print("\n".join(res["tail"]), flush=True)
summary["total_minutes"] = round((time.time() - t0) / 60, 1)
json.dump(summary, open("/kaggle/working/summary.json", "w"), indent=1)
print("SUMMARY", json.dumps(summary), flush=True)
