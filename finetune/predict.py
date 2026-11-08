"""Score saved LoRA + head weights on train.csv rows outside the lockbox (local GPU, after training ends).

Run: .venv/Scripts/python.exe finetune/predict.py --run run2 checkpoints/run2/keep/step1300.pt ...
Writes checkpoints/<run>/pred/<weights stem>.npy (logits, rows = val_ids.csv order) and val_ids.csv.
The model is built exactly as in train.py, from the run's resolved settings stored in its checkpoint.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("weights", nargs="+")
cli = ap.parse_args()
R = ROOT / "checkpoints" / cli.run
a = torch.load(R / "ckpt_latest" / "state.pt", map_location="cpu", weights_only=False)["args"]
MODEL = str(ROOT / "models/qwen3-4b-base")
classes = json.load(open(R / "classes.json"))

val = pd.read_csv(ROOT / "data/train.csv")
lock = list(GroupKFold(n_splits=a.get("cv_folds", 5)).split(val, val.model, val.prompt_id))[a.get("lockbox_fold", 4)][1]
val = val.drop(index=val.index[lock]).reset_index(drop=True)  # the lockbox is never scored here
y = val.model.map({c: i for i, c in enumerate(classes)}).to_numpy()
(R / "pred").mkdir(exist_ok=True)
val[["id", "prompt_id", "model"]].to_csv(R / "pred" / "val_ids.csv", index=False)

CDT = torch.bfloat16 if a.get("compute_dtype", "bf16") == "bf16" else torch.float16
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
cfg = AutoConfig.from_pretrained(MODEL)
cfg.num_hidden_layers = a["layers"] or cfg.num_hidden_layers
if getattr(cfg, "layer_types", None):
    cfg.layer_types = cfg.layer_types[:cfg.num_hidden_layers]
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=CDT)
base = AutoModel.from_pretrained(MODEL, config=cfg, quantization_config=bnb, dtype=CDT, device_map={"": 0})
base.norm = torch.nn.Identity()
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
base = get_peft_model(base, LoraConfig(r=a["lora_r"], lora_alpha=2 * a["lora_r"], lora_dropout=0.0,
                                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                       "up_proj", "down_proj"]))
depths = [int(x) for x in a["depths"].split(",")]


class MultiScale(torch.nn.Module):  # identical to train.py
    def __init__(self, base, depths, hidden, n_classes):
        super().__init__()
        self.base, self.depths = base, depths
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden) for _ in depths])
        self.drop = torch.nn.Dropout(0.1)
        self.head = torch.nn.Linear(hidden * len(depths), n_classes)

    def forward(self, input_ids, attention_mask):
        hs = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states
        m = attention_mask.unsqueeze(-1).float()
        denom = m.sum(1).clamp(min=1.0)
        return self.head(self.drop(torch.cat(
            [n((hs[d].float() * m).sum(1) / denom) for n, d in zip(self.norms, self.depths)], -1)))


model = MultiScale(base, depths, cfg.hidden_size, len(classes)).cuda().eval()
ids = tok(list(val.text.fillna("").astype(str)), truncation=True, max_length=a.get("eval_maxlen", 384),
          add_special_tokens=False)["input_ids"]
order = np.argsort([len(x) for x in ids], kind="stable")

for w in cli.weights:
    sd = torch.load(w, map_location="cpu", weights_only=False)
    res = model.load_state_dict(sd.get("trainable", sd), strict=False)
    assert not res.unexpected_keys, res.unexpected_keys[:5]
    out = np.zeros((len(ids), len(classes)), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(order), 16):
            b = order[s:s + 16]
            L = max(len(ids[i]) for i in b)
            x = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
            m = torch.zeros((len(b), L), dtype=torch.long)
            for r, i in enumerate(b):
                x[r, :len(ids[i])] = torch.tensor(ids[i])
                m[r, :len(ids[i])] = 1
            out[b] = model(x.cuda(), m.cuda()).float().cpu().numpy()
    np.save(R / "pred" / f"{Path(w).stem}.npy", out)
    print(f"{w}: raw {balanced_accuracy_score(y, out.argmax(1)):.4f} on {len(y)} rows", flush=True)
