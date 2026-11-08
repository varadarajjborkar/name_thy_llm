"""Resumable QLoRA fine-tune of Qwen3-4B-base as an 18-way LLM-authorship classifier.

Design goals: survive power cuts, crashes and pauses without losing more than one checkpoint interval.
  * Batch order is a pure function of (seed, epoch), so resuming only needs the step counter.
  * Checkpoints are written to a temp dir and swapped in atomically; the previous one is kept as a fallback.
  * Pause: create <run>/PAUSE (or Ctrl+C). The script finishes the step, saves, exits with code 10.
  * Trained on external data only. Validation is the competition train.csv (never trained on).

Normally launched through ctl.py. Direct use:
  .venv/Scripts/python.exe finetune/train.py --run run1
Exit codes: 0 finished, 10 paused, anything else = crash (ctl.py supervisor retries from the last checkpoint).
"""
import argparse
import ctypes
import json
import math
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import balanced_accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/qwen3-4b-base"
EXIT_PAUSED = 10

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="run1")
ap.add_argument("--train", default=str(ROOT / "data/processed/train_100k.parquet"))
ap.add_argument("--val", default=str(ROOT / "data/train.csv"))
ap.add_argument("--n-train", type=int, default=0, help="use only the first N rows (0 = all)")
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--head-lr", type=float, default=5e-4)
ap.add_argument("--warmup", type=float, default=0.03)
ap.add_argument("--maxlen", type=int, default=320)
ap.add_argument("--eval-maxlen", type=int, default=384)
ap.add_argument("--ckpt-every", type=int, default=100, help="optimizer steps")
ap.add_argument("--eval-every", type=int, default=500, help="optimizer steps")
ap.add_argument("--eval-n", type=int, default=900, help="stratified val rows for periodic eval")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--lora-r", type=int, default=16)
a = ap.parse_args()

RUN = ROOT / "checkpoints" / a.run
RUN.mkdir(parents=True, exist_ok=True)
LOGF = open(RUN / "train.log", "a", buffering=1, encoding="utf-8")


def P(*s):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in s)
    print(msg, flush=True)
    LOGF.write(msg + "\n")


# keep Windows awake while training (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
try:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
except Exception:
    pass

stop = {"flag": False}
for sig in (signal.SIGINT, getattr(signal, "SIGBREAK", signal.SIGINT)):
    signal.signal(sig, lambda *_: stop.__setitem__("flag", True))

if (RUN / "DONE_TRAINING").exists():
    P("already finished; nothing to do")
    sys.exit(0)

# ------------------------------------------------------------------ data
val = pd.read_csv(a.val)
classes = sorted(val.model.unique())
cid = {c: i for i, c in enumerate(classes)}
tr = pd.read_parquet(a.train, columns=["text", "model"])
if a.n_train:
    tr = tr.iloc[: a.n_train]
assert set(tr.model) <= set(classes), "label mismatch between train and val"

(RUN / "classes.json").write_text(json.dumps(classes))
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
PAD = tok.pad_token_id


def encode(texts, maxlen):
    ids = tok(list(texts), truncation=True, max_length=maxlen, add_special_tokens=False)["input_ids"]
    return [np.asarray(x, dtype=np.int32) for x in ids]


t0 = time.time()
tr_ids = encode(tr.text, a.maxlen)
tr_len = np.array([len(x) for x in tr_ids])
tr_y = tr.model.map(cid).to_numpy()
# fixed stratified subset of val for periodic eval; full val at the end
per_class = max(1, a.eval_n // len(classes))
sub_idx = val.groupby("model").sample(n=per_class, random_state=0).index.to_numpy()
val_ids = encode(val.text, a.eval_maxlen)
val_y = val.model.map(cid).to_numpy()
P(f"data: train {len(tr_ids):,} val {len(val_ids):,} (periodic eval on {len(sub_idx)}) tokenized in {time.time() - t0:.0f}s; mean len {tr_len.mean():.0f}")


def epoch_plan(epoch):
    """Deterministic length-bucketed batches: shuffle, sort inside big chunks, then shuffle the batches."""
    rng = np.random.default_rng(a.seed + epoch)
    perm = rng.permutation(len(tr_ids))
    chunk = a.bs * a.accum * 32
    batches = []
    for s in range(0, len(perm), chunk):
        c = perm[s:s + chunk]
        c = c[np.argsort(tr_len[c], kind="stable")]
        batches += [c[b:b + a.bs] for b in range(0, len(c), a.bs)]
    return [batches[i] for i in rng.permutation(len(batches))]


steps_per_epoch = len(epoch_plan(0)) // a.accum
total_steps = steps_per_epoch * a.epochs
P(f"plan: {steps_per_epoch} steps/epoch x {a.epochs} = {total_steps} optimizer steps, effective batch {a.bs * a.accum}")


def collate(ids_list, idx):
    seqs = [ids_list[i] for i in idx]
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), PAD, dtype=torch.long)
    m = torch.zeros((len(seqs), L), dtype=torch.long)
    for r, s in enumerate(seqs):
        x[r, : len(s)] = torch.from_numpy(s)
        m[r, : len(s)] = 1
    return x.cuda(non_blocking=True), m.cuda(non_blocking=True)


# ------------------------------------------------------------------ model
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
torch.manual_seed(a.seed)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(classes), quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0})
model.config.pad_token_id = PAD
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(
    r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, task_type="SEQ_CLS",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
head = [p for n, p in named if "score" in n]
body = [p for n, p in named if "score" not in n]
assert head and body
P("trainable dtypes:", sorted({str(p.dtype) for _, p in named}), f"({sum(p.numel() for _, p in named) / 1e6:.1f}M params)")
opt = torch.optim.AdamW([{"params": body, "lr": a.lr, "base": a.lr},
                         {"params": head, "lr": a.head_lr, "base": a.head_lr}], weight_decay=0.01)


def lr_mult(step):
    w = max(1, int(a.warmup * total_steps))
    if step < w:
        return (step + 1) / w
    prog = (step - w) / max(1, total_steps - w)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))


# ------------------------------------------------------------------ checkpoints
LATEST, PREV, TMP = RUN / "ckpt_latest", RUN / "ckpt_prev", RUN / "ckpt_tmp"


def save_ckpt(step, best):
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir()
    with open(TMP / "state.pt", "wb") as fh:
        torch.save({"trainable": {n: p.detach().cpu() for n, p in named}, "opt": opt.state_dict(),
                    "step": step, "best": best, "args": vars(a)}, fh)
        fh.flush()
        os.fsync(fh.fileno())  # data on disk before we declare it complete
    with open(TMP / "OK", "w") as fh:
        fh.write(str(step))
        fh.flush()
        os.fsync(fh.fileno())
    if PREV.exists():
        shutil.rmtree(PREV)
    if LATEST.exists():
        os.replace(LATEST, PREV)
    os.replace(TMP, LATEST)
    (RUN / "progress.json").write_text(json.dumps({"step": step, "total": total_steps, "best": best}))


def load_ckpt():
    # TMP first: it only has an OK marker if a crash hit between the two swaps, and then it is the newest.
    for d in (TMP, LATEST, PREV):
        if (d / "OK").exists():
            try:
                st = torch.load(d / "state.pt", map_location="cpu", weights_only=False)
                for k in ("seed", "bs", "accum", "epochs", "n_train", "lora_r", "maxlen", "train", "lr", "head_lr", "warmup"):
                    if st["args"].get(k) != getattr(a, k):
                        P(f"FATAL: --{k} changed since the checkpoint ({st['args'].get(k)} -> {getattr(a, k)}). "
                          f"Resuming would silently use a different batch order/schedule. Use the old value or a new --run.")
                        sys.exit(4)
                bad = model.load_state_dict(st["trainable"], strict=False).unexpected_keys
                assert not bad, bad
                opt.load_state_dict(st["opt"])
                P(f"resumed from {d.name} at step {st['step']}")
                return st["step"], st["best"]
            except SystemExit:
                raise
            except Exception as e:  # corrupt checkpoint: fall back to the previous one
                P(f"checkpoint {d.name} unusable ({e!r}), trying older")
    if any((d / "OK").exists() for d in (TMP, LATEST, PREV)):
        P("FATAL: checkpoints exist but none could be loaded. Refusing to restart from step 0 "
          "(that would overwrite them). Inspect the log above, or move the checkpoints folder away on purpose.")
        sys.exit(4)
    return 0, 0.0


step, best = load_ckpt()


# ------------------------------------------------------------------ eval
@torch.no_grad()
def evaluate(idx, bs=16):
    model.eval()
    order = idx[np.argsort([len(val_ids[i]) for i in idx], kind="stable")]
    out = np.zeros((len(val_ids), len(classes)), dtype=np.float32)
    for s in range(0, len(order), bs):
        b = order[s:s + bs]
        x, m = collate(val_ids, b)
        out[b] = model(input_ids=x, attention_mask=m).logits.float().cpu().numpy()
    model.train()
    return out


def bal_acc(logits, idx):
    return balanced_accuracy_score(val_y[idx], logits[idx].argmax(1))


def recall_report(logits, idx):
    """Per-model recall (share of each model's passages identified) and the full confusion matrix.
    cm[i][j] = passages written by model i that were predicted as model j. Mean recall = balanced accuracy."""
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(cm, (val_y[idx], logits[idx].argmax(1)), 1)
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    return [round(float(r), 4) for r in recall], cm.tolist()


# ------------------------------------------------------------------ train
model.train()
opt.zero_grad(set_to_none=True)
micro_per_epoch = steps_per_epoch * a.accum
plans = {}
ema, nonfinite, t_start, step0 = None, 0, time.time(), step
P(f"start at step {step}/{total_steps}")

while step < total_steps:
    ep = step // steps_per_epoch
    if ep not in plans:
        plans = {ep: epoch_plan(ep)}
    plan = plans[ep]
    base = (step % steps_per_epoch) * a.accum
    for g in opt.param_groups:
        g["lr"] = g["base"] * lr_mult(step)
    tot = 0.0
    for j in range(a.accum):
        idx = plan[base + j]
        x, m = collate(tr_ids, idx)
        y = torch.tensor(tr_y[idx], device="cuda")
        loss = torch.nn.functional.cross_entropy(model(input_ids=x, attention_mask=m).logits.float(), y)
        if not torch.isfinite(loss):
            nonfinite += 1
            if nonfinite > 20:
                P("too many non-finite losses, aborting")
                sys.exit(3)
            continue
        (loss / a.accum).backward()
        tot += loss.item() / a.accum
    torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    step += 1
    ema = tot if ema is None else 0.98 * ema + 0.02 * tot

    if step % 20 == 0:
        done = step - step0
        eta = (time.time() - t_start) / done * (total_steps - step) / 3600
        P(f"step {step}/{total_steps} loss {ema:.3f} lr {opt.param_groups[0]['lr']:.2e} eta {eta:.1f}h")

    if step % a.eval_every == 0 or step == total_steps:
        lg = evaluate(sub_idx)
        acc = bal_acc(lg, sub_idx)
        P(f"EVAL step {step} balanced_acc {acc:.4f} (subset {len(sub_idx)})")
        rec, cm = recall_report(lg, sub_idx)
        with open(RUN / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"step": step, "loss": ema, "val_bal_acc": acc, "t": time.time(),
                                "recall": rec, "confusion": cm}) + "\n")
        if acc > best:
            best = acc
            torch.save({n: p.detach().cpu() for n, p in named}, RUN / "best_tmp.pt")
            os.replace(RUN / "best_tmp.pt", RUN / "best_trainable.pt")

    paused = stop["flag"] or (RUN / "PAUSE").exists()
    if step % a.ckpt_every == 0 or paused or step == total_steps:
        save_ckpt(step, best)
        P(f"checkpoint saved at step {step}")
    if paused and step < total_steps:
        P("pause requested: state saved, exiting")
        sys.exit(EXIT_PAUSED)

# ------------------------------------------------------------------ finish
model.save_pretrained(RUN / "final")
lg = evaluate(np.arange(len(val_ids)))
acc = bal_acc(lg, np.arange(len(val_ids)))
np.save(RUN / "val_logits.npy", lg)
val[["id", "prompt_id", "model"]].to_csv(RUN / "val_ids.csv", index=False)
P(f"FINAL full-val balanced_acc {acc:.4f}; adapter in {RUN / 'final'}; logits in val_logits.npy")
rec, cm = recall_report(lg, np.arange(len(val_ids)))
(RUN / "final_eval.json").write_text(json.dumps({"bal_acc": acc, "classes": classes, "recall": rec, "confusion": cm}))
(RUN / "DONE_TRAINING").write_text(f"{acc:.4f}")
