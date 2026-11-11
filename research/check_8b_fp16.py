"""Does an 8B member give the same predictions under the notebook's inference settings (4-bit nf4 with double
quant, fp16 compute, as on a T4) as in training (bf16)? Scores N fold-0 rows with checkpoints/k8f0 weights, built the
same way as kernel_qwen/ntl_qwen_blend.py qwen_scores(), and compares with the logits saved by the trainer.
Run: .venv/Scripts/python.exe research/check_8b_fp16.py [N]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "checkpoints/k8f0"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
MODEL = str(ROOT / "models/qwen3-8b-base")
args = json.load(open(R / "run_args.json"))
classes = json.load(open(R / "classes.json"))
ids = pd.read_csv(R / "val_ids.csv").id.astype(str).to_numpy()[:N]
ref = np.load(R / "val_logits.npy")[:N]
train = pd.read_csv(ROOT / "data/train.csv")
train["id"] = train.id.astype(str)
text = train.set_index("id").loc[ids, "text"].fillna("").astype(str).tolist()

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.pad_token or tok.eos_token
cfg = AutoConfig.from_pretrained(MODEL)
cfg.num_hidden_layers = args["layers"] or cfg.num_hidden_layers
if getattr(cfg, "layer_types", None):
    cfg.layer_types = cfg.layer_types[:cfg.num_hidden_layers]
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.float16)
base = AutoModel.from_pretrained(MODEL, config=cfg, quantization_config=bnb, dtype=torch.float16, device_map={"": 0})
base.norm = torch.nn.Identity()
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
base = get_peft_model(base, LoraConfig(r=args["lora_r"], lora_alpha=2 * args["lora_r"], lora_dropout=0.0,
                                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                       "down_proj"]))
depths = [int(x) for x in args["depths"].split(",")]


class MultiScale(torch.nn.Module):  # same as the notebook
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
        feats = [norm((hs[d].float() * m).sum(1) / denom) for norm, d in zip(self.norms, self.depths)]
        return self.head(self.drop(torch.cat(feats, -1)))


model = MultiScale(base, depths, cfg.hidden_size, len(classes)).cuda().eval()
res = model.load_state_dict(torch.load(R / "final_trainable.pt", map_location="cpu", weights_only=False), strict=False)
assert not res.unexpected_keys and not [k for k in res.missing_keys if "lora_" in k or k.startswith(("norms.", "head."))]
enc = tok(text, truncation=True, max_length=args.get("eval_maxlen", 384), add_special_tokens=False)["input_ids"]
out = np.zeros((N, len(classes)), dtype=np.float32)
with torch.no_grad():
    for s in range(0, N, 16):
        b = list(range(s, min(s + 16, N)))
        L = max(len(enc[i]) for i in b)
        x = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
        m = torch.zeros((len(b), L), dtype=torch.long)
        for r, i in enumerate(b):
            x[r, :len(enc[i])] = torch.tensor(enc[i])
            m[r, :len(enc[i])] = 1
        out[b] = model(x.cuda(), m.cuda()).float().cpu().numpy()
assert np.isfinite(out).all(), "non-finite fp16 logits"
y = pd.Series(ids).map(train.set_index("id").model).map({c: i for i, c in enumerate(classes)}).to_numpy()
agree = (out.argmax(1) == ref.argmax(1)).mean()
corr = np.corrcoef(out.ravel(), ref.ravel())[0, 1]
print(f"rows {N} | argmax agreement fp16 vs training {agree:.3f} | logit corr {corr:.4f} | "
      f"max abs diff {np.abs(out - ref).max():.3f} | acc fp16 {(out.argmax(1) == y).mean():.3f} "
      f"vs training {(ref.argmax(1) == y).mean():.3f}")
