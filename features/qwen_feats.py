"""Extract multi-scale Qwen features (the input of the classifier head) for the duel-specialist experiments.

Features = LayerNorm'd mean-pooled hidden states at the 8 pooled depths, concatenated (8 x 2560 = 20,480 dims),
exactly what the trained head sees. Weights: a stage-1 run (default run3) that never trained on host data, so
features of host rows are honest for out-of-fold tests.
Rows:
  host  train.csv minus the lockbox fold 4 (2,160 rows)
  pool  sibling-group models only, pool rows that neither run2 nor run3 trained on, up to --per-class each
Output (gitignored): data/processed/feats_<run>_{host,pool}.npy (float16) + matching .parquet row tables.
Run: .venv/Scripts/python.exe features/qwen_feats.py --run run3 --per-class 1500
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import GroupKFold
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune"))
from prep_data import typography_mask  # noqa: E402

GROUPS = {
    "glm": ["glm-5.3", "glm-5.3-flash"],
    "gemini_flash": ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"],
    "gpt56": ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"],
    "claude": ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5"],
}
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="run3")
ap.add_argument("--per-class", type=int, default=1500)
cli = ap.parse_args()
R = ROOT / "checkpoints" / cli.run
a = torch.load(R / "ckpt_latest" / "state.pt", map_location="cpu", weights_only=False)["args"]
MODEL = a.get("model") or str(ROOT / "models/qwen3-4b-base")
classes = json.load(open(R / "classes.json"))
OUT = ROOT / "data/processed"

host = pd.read_csv(ROOT / "data/train.csv")
lock = list(GroupKFold(5).split(host, host.model, host.prompt_id))[4][1]
host = host.drop(index=host.index[lock]).reset_index(drop=True)
used = set(pd.read_parquet(OUT / "pool_typo_60k.parquet", columns=["text"]).text) | \
    set(pd.read_parquet(OUT / "pool_cont_30k.parquet", columns=["text"]).text)
pool = pd.read_parquet(OUT / "reserve_typo.parquet", columns=["text", "model"])
pool = pool[typography_mask(pool) & ~pool.text.isin(used) & ~pool.text.isin(set(host.text))]
sib = sorted({m for g in GROUPS.values() for m in g})
pool = pd.concat([g.sample(n=min(len(g), cli.per_class), random_state=3)
                  for m, g in pool[pool.model.isin(sib)].groupby("model")]).reset_index(drop=True)
print(f"host {len(host)} rows, pool {len(pool)} rows ({pool.model.value_counts().min()}-{pool.model.value_counts().max()} per class)")

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


class MultiScale(torch.nn.Module):  # identical to train.py; features() returns what the head receives
    def __init__(self, base, depths, hidden, n_classes):
        super().__init__()
        self.base, self.depths = base, depths
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden) for _ in depths])
        self.drop = torch.nn.Dropout(0.1)
        self.head = torch.nn.Linear(hidden * len(depths), n_classes)

    def features(self, input_ids, attention_mask):
        hs = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states
        m = attention_mask.unsqueeze(-1).float()
        denom = m.sum(1).clamp(min=1.0)
        return torch.cat([n((hs[d].float() * m).sum(1) / denom) for n, d in zip(self.norms, self.depths)], -1)


model = MultiScale(base, depths, cfg.hidden_size, len(classes)).cuda().eval()
sd = torch.load(R / "final_trainable.pt", map_location="cpu", weights_only=False)
assert not model.load_state_dict(sd, strict=False).unexpected_keys


@torch.no_grad()
def feats(texts):
    ids = tok(list(texts), truncation=True, max_length=a.get("eval_maxlen", 384), add_special_tokens=False)["input_ids"]
    order = np.argsort([len(x) for x in ids], kind="stable")
    out = np.zeros((len(ids), cfg.hidden_size * len(depths)), dtype=np.float16)
    for s in range(0, len(order), 16):
        b = order[s:s + 16]
        L = max(len(ids[i]) for i in b)
        x = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
        m = torch.zeros((len(b), L), dtype=torch.long)
        for r, i in enumerate(b):
            x[r, :len(ids[i])] = torch.tensor(ids[i])
            m[r, :len(ids[i])] = 1
        out[b] = model.features(x.cuda(), m.cuda()).cpu().numpy().astype(np.float16)
        if s % 1600 == 0:
            print(f"  {s}/{len(order)}", flush=True)
    return out


for name, df in (("host", host[["id", "prompt_id", "model", "text"]]), ("pool", pool)):
    np.save(OUT / f"feats_{cli.run}_{name}.npy", feats(df.text.fillna("").astype(str)))
    df.drop(columns=["text"]).to_parquet(OUT / f"feats_{cli.run}_{name}.parquet", index=False)
    print(f"saved {name}", flush=True)
