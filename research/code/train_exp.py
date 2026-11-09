"""Resumable QLoRA fine-tune of Qwen3-4B-base as an 18-way LLM-authorship classifier.

Design goals: survive power cuts, crashes and pauses without losing more than one checkpoint interval.
  * Batch order is a pure function of (seed, epoch), so resuming only needs the step counter.
  * Checkpoints are written to a temp dir and swapped in atomically; the previous one is kept as a fallback.
  * Pause: create <run>/PAUSE (or Ctrl+C). The script finishes the step, saves, exits with code 10.
  * Trained on external data only. Validation is the competition train.csv (never trained on).

Two architectures:
  --arch seqcls   Qwen's built-in sequence-classification head (last token, all layers). Used by run1.
  --arch ms       "multi-scale": keep only the first --layers transformer layers (style lives low and mid,
                  the top layers mostly encode topic), mean-pool the hidden states at several depths, LayerNorm
                  each, concatenate, linear head. Follows the public "LoRA24" recipe for this competition.
Two stages:
  stage 1  train on the external pool, validate on the competition train.csv
  stage 2  --init <stage-1 weights> --cv-fold k: fine-tune on train.csv minus fold k, validate on fold k
           (GroupKFold by prompt_id), or --cv-fold -2 to train on all of train.csv for the submission.

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
from sklearn.model_selection import GroupKFold
from transformers import (AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
                          BitsAndBytesConfig)

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
ap.add_argument("--arch", choices=["seqcls", "ms"], default="seqcls")
ap.add_argument("--layers", type=int, default=0, help="ms only: keep the first N transformer layers (0 = all)")
ap.add_argument("--depths", default="2,4,6,8,12,16,20,24", help="ms only: layers whose hidden states are pooled")
ap.add_argument("--init", default="", help="start from these trainable weights (a .pt file or a run's checkpoint dir)")
ap.add_argument("--cv-fold", type=int, default=-1,
                help="-1: train on --train, validate on --val. k>=0: train on --val minus fold k, validate on fold k. "
                     "-2: train on all of --val (no validation)")
ap.add_argument("--cv-folds", type=int, default=5)
ap.add_argument("--label-smoothing", type=float, default=0.0)
ap.add_argument("--compute-dtype", choices=["bf16", "fp16"], default="bf16",
                help="fp16 for GPUs without native bf16 (Kaggle T4/P100); uses loss scaling")
ap.add_argument("--extra", default="", help="parquet (text, model) appended to the stage-2 training rows (never to validation)")
ap.add_argument("--extra-n", type=int, default=0, help="use a random sample of N extra rows (0 = all)")
ap.add_argument("--crop-min", type=float, default=0.0,
                help="train-time augmentation: each passage is cut to a random contiguous span of [crop-min, 1] of its tokens (0 = off)")
ap.add_argument("--lora-dropout", type=float, default=0.05)
ap.add_argument("--head-dropout", type=float, default=0.1)
ap.add_argument("--lockbox-fold", type=int, default=-1,
                help="stage 1: drop this GroupKFold fold of --val from validation, so no decision ever looks at it")
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
if a.cv_fold == -1:
    tr = pd.read_parquet(a.train, columns=["text", "model"])
    if a.lockbox_fold >= 0:  # the lockbox stays unseen until the final, one-time check
        lock_rows = list(GroupKFold(n_splits=a.cv_folds).split(val, val.model, val.prompt_id))[a.lockbox_fold][1]
        val = val.drop(index=val.index[lock_rows]).reset_index(drop=True)
elif a.cv_fold == -2:  # final stage 2: all host data, keep a tiny copy as a smoke check only
    tr = val[["text", "model"]].copy()
else:  # stage 2 cross-validation: folds grouped by prompt so no topic is shared between train and validation
    folds = list(GroupKFold(n_splits=a.cv_folds).split(val, val.model, val.prompt_id))
    tr_rows, va_rows = folds[a.cv_fold]
    tr = val.iloc[tr_rows][["text", "model"]].reset_index(drop=True)
    val = val.iloc[va_rows].reset_index(drop=True)
if a.extra:
    ex = pd.read_parquet(a.extra, columns=["text", "model"])
    if a.extra_n:
        ex = ex.sample(n=min(a.extra_n, len(ex)), random_state=a.seed)
    tr = pd.concat([tr, ex], ignore_index=True)
if a.n_train:
    tr = tr.iloc[: a.n_train]
assert set(tr.model) <= set(classes), "label mismatch between train and val"

(RUN / "classes.json").write_text(json.dumps(classes))
(RUN / "run_args.json").write_text(json.dumps(vars(a)))  # resolved settings, needed to rebuild the model for inference
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
if a.eval_n and a.eval_n < len(val):
    per_class = max(1, a.eval_n // len(classes))
    sub_idx = val.groupby("model").sample(n=per_class, random_state=0).index.to_numpy()
else:  # --eval-n 0: evaluate on every validation row
    sub_idx = np.arange(len(val))
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


def crop(s, rng):
    if a.crop_min <= 0 or len(s) < 48:
        return s
    n = int(len(s) * rng.uniform(a.crop_min, 1.0))
    st = int(rng.integers(0, len(s) - n + 1))
    return s[st:st + n]


def collate(ids_list, idx, rng=None):
    seqs = [ids_list[i] for i in idx]
    if rng is not None:
        seqs = [crop(s, rng) for s in seqs]
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), PAD, dtype=torch.long)
    m = torch.zeros((len(seqs), L), dtype=torch.long)
    for r, s in enumerate(seqs):
        x[r, : len(s)] = torch.from_numpy(s)
        m[r, : len(s)] = 1
    return x.cuda(non_blocking=True), m.cuda(non_blocking=True)


# ------------------------------------------------------------------ model
CDT = torch.bfloat16 if a.compute_dtype == "bf16" else torch.float16
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=CDT)
scaler = torch.amp.GradScaler("cuda", enabled=a.compute_dtype == "fp16")  # pass-through when bf16
torch.manual_seed(a.seed)
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


HEAD_DROPOUT = a.head_dropout


class MultiScale(torch.nn.Module):
    """Mean-pooled hidden states at several depths -> LayerNorm each -> concat -> dropout -> linear."""

    def __init__(self, base, depths, hidden, n_classes):
        super().__init__()
        self.base, self.depths = base, depths
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden) for _ in depths])
        self.drop = torch.nn.Dropout(HEAD_DROPOUT)
        self.head = torch.nn.Linear(hidden * len(depths), n_classes)

    def forward(self, input_ids, attention_mask):
        hs = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states
        m = attention_mask.unsqueeze(-1).float()
        denom = m.sum(1).clamp(min=1.0)
        feats = [norm((hs[d].float() * m).sum(1) / denom) for norm, d in zip(self.norms, self.depths)]
        return self.head(self.drop(torch.cat(feats, -1)))


if a.arch == "seqcls":
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(classes), quantization_config=bnb, dtype=CDT, device_map={"": 0})
    model.config.pad_token_id = PAD
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=a.lora_dropout,
                                             task_type="SEQ_CLS", target_modules=LORA_TARGETS))
else:
    cfg = AutoConfig.from_pretrained(MODEL)
    n_layers = a.layers or cfg.num_hidden_layers
    depths = [int(x) for x in a.depths.split(",")]
    assert max(depths) <= n_layers and min(depths) >= 1, f"--depths must lie in 1..{n_layers}"
    cfg.num_hidden_layers = n_layers  # the unused top layers are never even loaded
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = cfg.layer_types[:n_layers]
    base = AutoModel.from_pretrained(MODEL, config=cfg, quantization_config=bnb, dtype=CDT,
                                     device_map={"": 0})
    base.norm = torch.nn.Identity()  # the deepest pooled layer is read raw, like the shallower ones
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    base = get_peft_model(base, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=a.lora_dropout,
                                           target_modules=LORA_TARGETS))
    model = MultiScale(base, depths, cfg.hidden_size, len(classes)).cuda()
    P(f"multi-scale: {n_layers} layers, pooling depths {depths}")


def logits_of(x, m):
    out = model(input_ids=x, attention_mask=m)
    return out.logits if hasattr(out, "logits") else out


named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
is_head = lambda n: "score" in n or n.startswith(("norms.", "head."))
head = [p for n, p in named if is_head(n)]
body = [p for n, p in named if not is_head(n)]
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
                    "step": step, "best": best, "args": vars(a), "ema": ema}, fh)
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
                for k in ("seed", "bs", "accum", "epochs", "n_train", "lora_r", "maxlen", "train", "lr", "head_lr",
                          "warmup", "arch", "layers", "depths", "cv_fold", "cv_folds", "label_smoothing", "init",
                          "lockbox_fold", "compute_dtype"):
                    old = st["args"].get(k, ap.get_default(k))  # checkpoints from before an option existed
                    if old != getattr(a, k):
                        P(f"FATAL: --{k.replace('_', '-')} changed since the checkpoint ({old} -> {getattr(a, k)}). "
                          f"Resuming would silently use a different batch order/schedule. Use the old value or a new --run.")
                        sys.exit(4)
                bad = model.load_state_dict(st["trainable"], strict=False).unexpected_keys
                assert not bad, bad
                opt.load_state_dict(st["opt"])
                P(f"resumed from {d.name} at step {st['step']}")
                return st["step"], st["best"], st.get("ema")
            except SystemExit:
                raise
            except Exception as e:  # corrupt checkpoint: fall back to the previous one
                P(f"checkpoint {d.name} unusable ({e!r}), trying older")
    if any((d / "OK").exists() for d in (TMP, LATEST, PREV)):
        P("FATAL: checkpoints exist but none could be loaded. Refusing to restart from step 0 "
          "(that would overwrite them). Inspect the log above, or move the checkpoints folder away on purpose.")
        sys.exit(4)
    return 0, 0.0, None


ema = None
step, best, ema = load_ckpt()
if step == 0 and a.init:  # stage 2: start from stage-1 weights (fresh optimizer and schedule)
    src = Path(a.init)
    src = src / "ckpt_latest" / "state.pt" if src.is_dir() else src
    sd = torch.load(src, map_location="cpu", weights_only=False)
    sd = sd.get("trainable", sd)
    res = model.load_state_dict(sd, strict=False)
    own = {n for n, _ in named}
    missing = [k for k in res.missing_keys if k in own]
    assert not res.unexpected_keys and not missing, (res.unexpected_keys[:5], missing[:5])
    P(f"initialised {len(sd)} trainable tensors from {src}")


# ------------------------------------------------------------------ eval
@torch.no_grad()
def evaluate(idx, bs=16):
    model.eval()
    order = idx[np.argsort([len(val_ids[i]) for i in idx], kind="stable")]
    out = np.zeros((len(val_ids), len(classes)), dtype=np.float32)
    for s in range(0, len(order), bs):
        b = order[s:s + bs]
        x, m = collate(val_ids, b)
        out[b] = logits_of(x, m).float().cpu().numpy()
    model.train()
    return out


def bal_acc(logits, idx):
    return balanced_accuracy_score(val_y[idx], logits[idx].argmax(1))


def sinkhorn(logits, iters=100):
    """Class balancing without labels: rescale probabilities so every class gets an equal share of the predictions
    (rows renormalised each round). Assumes the evaluated set is roughly balanced, as the host's train.csv is."""
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    for _ in range(iters):
        p /= p.sum(0, keepdims=True)
        p /= p.sum(1, keepdims=True)
    return np.log(p + 1e-12)


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
nonfinite, t_start, step0 = 0, time.time(), step  # ema comes from the checkpoint, so a resume does not reset it
P(f"start at step {step}/{total_steps}")

while step < total_steps:
    ep = step // steps_per_epoch
    if ep not in plans:
        plans = {ep: epoch_plan(ep)}
    plan = plans[ep]
    first_micro = (step % steps_per_epoch) * a.accum
    for g in opt.param_groups:
        g["lr"] = g["base"] * lr_mult(step)
    tot = 0.0
    for j in range(a.accum):
        idx = plan[first_micro + j]
        x, m = collate(tr_ids, idx, np.random.default_rng((a.seed, step, j)))
        y = torch.tensor(tr_y[idx], device="cuda")
        loss = torch.nn.functional.cross_entropy(logits_of(x, m).float(), y, label_smoothing=a.label_smoothing)
        if not torch.isfinite(loss):
            nonfinite += 1
            if nonfinite > 20:
                P("too many non-finite losses, aborting")
                sys.exit(3)
            continue
        scaler.scale(loss / a.accum).backward()
        tot += loss.item() / a.accum
    scaler.unscale_(opt)  # no-ops when bf16; with fp16 a step with inf/NaN gradients is skipped, not applied
    torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0)
    scaler.step(opt)
    scaler.update()
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
        lg_b = lg.copy()
        lg_b[sub_idx] = sinkhorn(lg[sub_idx])
        acc_b = bal_acc(lg_b, sub_idx)
        P(f"EVAL step {step} balanced_acc {acc:.4f} (subset {len(sub_idx)}) | with class balancing {acc_b:.4f}")
        rec, cm = recall_report(lg, sub_idx)
        rec_b, cm_b = recall_report(lg_b, sub_idx)
        (RUN / "eval_logits").mkdir(exist_ok=True)  # raw outputs kept so any post-processing can be checked later
        np.save(RUN / "eval_logits" / f"step{step}.npy", lg[sub_idx])
        np.save(RUN / "eval_logits" / "labels.npy", val_y[sub_idx])
        with open(RUN / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"step": step, "loss": ema, "val_bal_acc": acc, "t": time.time(),
                                "recall": rec, "confusion": cm, "val_bal_acc_balanced": acc_b,
                                "recall_balanced": rec_b, "confusion_balanced": cm_b}) + "\n")
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
torch.save({n: p.detach().cpu() for n, p in named}, RUN / "final_trainable.pt")  # LoRA + head, for stage 2 / inference
if a.arch == "seqcls":
    model.save_pretrained(RUN / "final")
lg = evaluate(np.arange(len(val_ids)))
acc = bal_acc(lg, np.arange(len(val_ids)))
np.save(RUN / "val_logits.npy", lg)
val[["id", "prompt_id", "model"]].to_csv(RUN / "val_ids.csv", index=False)
note = " (trained on these rows, not a real score)" if a.cv_fold == -2 else ""
P(f"FINAL full-val balanced_acc {acc:.4f}{note}; weights in final_trainable.pt; logits in val_logits.npy")
rec, cm = recall_report(lg, np.arange(len(val_ids)))
(RUN / "final_eval.json").write_text(json.dumps({"bal_acc": acc, "classes": classes, "recall": rec, "confusion": cm}))
(RUN / "DONE_TRAINING").write_text(f"{acc:.4f}")
