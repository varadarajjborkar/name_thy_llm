"""Measure real VRAM use and training speed for Qwen3-4B-base QLoRA classification.

Run:  .venv/Scripts/python.exe finetune/vram_test.py [--bs 8] [--maxlen 320] [--steps 15] [--no-gc]
Nothing is saved. It only reports token lengths, peak VRAM and samples/second.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/qwen3-4b-base"

ap = argparse.ArgumentParser()
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--maxlen", type=int, default=320)
ap.add_argument("--steps", type=int, default=15)
ap.add_argument("--no-gc", action="store_true", help="disable gradient checkpointing")
ap.add_argument("--sorted", action="store_true", help="length-bucketed batches (shuffled batch order)")
a = ap.parse_args()

df = pd.read_parquet(ROOT / "data/processed/train_100k.parquet", columns=["text", "model"]).sample(2000, random_state=0)
classes = sorted(df.model.unique())
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

lens = pd.Series([len(x) for x in tok(df.text.tolist())["input_ids"]])
print("token lengths:", lens.describe(percentiles=[0.5, 0.9, 0.99]).round(0).to_dict())
print(f"share truncated at maxlen={a.maxlen}: {(lens > a.maxlen).mean():.3%}")

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
t0 = time.time()
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=18, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0})
model.config.pad_token_id = tok.pad_token_id
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=not a.no_gc)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
print(f"load time {time.time() - t0:.0f}s   VRAM after load {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
model.train()
df = df.assign(_len=lens.values)
if a.sorted:
    import numpy as np
    df = df.sort_values("_len").reset_index(drop=True)
    order = np.random.default_rng(0).permutation(len(df) // a.bs)
    df = pd.concat([df.iloc[o * a.bs:(o + 1) * a.bs] for o in order]).reset_index(drop=True)
y_all = torch.tensor([classes.index(m) for m in df.model])
torch.cuda.reset_peak_memory_stats()
times = []
for step in range(a.steps):
    i = slice(step * a.bs, (step + 1) * a.bs)
    b = tok(df.text.iloc[i].tolist(), truncation=True, max_length=a.maxlen, padding=True, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    t = time.time()
    loss = model(**b, labels=y_all[i].cuda()).loss
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    times.append(time.time() - t)
    print(f"step {step:2d} loss {loss.item():.3f} {times[-1]:.2f}s seq {b['input_ids'].shape[1]}")

steady = times[3:]  # skip warm-up
sps = a.bs / (sum(steady) / len(steady))
print(f"\nPEAK VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB of 8   speed {sps:.1f} samples/s")
for n in (20_000, 50_000, 100_000):
    print(f"  one epoch over {n:,}: ~{n / sps / 3600:.1f} h")
