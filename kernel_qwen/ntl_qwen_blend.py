"""Name That LLM: fine-tuned Qwen3-4B (multi-scale LoRA) + TF-IDF SVM + LightGBM style blend. GPU, internet off.

Inputs (attached datasets):
  ntl-qwen-weights : one folder per Qwen member with weights.pt (LoRA + head), args.json, classes.json
  ntl-lgbm-style   : lgbm_ext.txt, style_features.py
  ntl-wheels       : bitsandbytes wheel (not preinstalled on Kaggle, no internet on the rerun)
  model qwen-lm/qwen-3/transformers/4b-base
Blend: sum_i w_i * log_softmax(T_i * scores_i), weights fixed in advance from out-of-fold predictions on train.csv.
Optional Sinkhorn class balancing at the end (the metric is balanced accuracy).
"""
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.environ.get("NTL_ROOT", "/kaggle/input")
# member -> (weight, temperature). Qwen members are listed by their folder name inside ntl-qwen-weights.
# History: stage-1 Qwen (final+s1700) 0.8 + SVM 0.1 + LGBM 0.1: OOF 0.543/0.546, LB 0.520 argmax / 0.521 sinkhorn.
# v3: stage-2 s2all 0.6 + stage-1 0.2 + SVM 0.1 + LGBM 0.1: OOF raw 0.575 / sinkhorn 0.580, LB 0.550.
# Now (research/fit_blend_multi.py, folds 0-3 OOF, 2,160 rows): three stage-2 full fits from different stage-1
# starts (run2, run3 = run2 continued on unseen pool rows, k_host3 = pool + hostlike x3), equal weights 0.8 in
# total + SVM 0.1 + LGBM 0.1 -> raw 0.589, sinkhorn 0.591. Stage 1 adds nothing on top.
WEIGHTS = {"qwen:s2all": (0.8 / 3, 1.0), "qwen:s3all": (0.8 / 3, 1.0), "qwen:h3s2all": (0.8 / 3, 1.0),
           "svm": (0.1, 3.0), "lgbm": (0.1, 1.0)}
SINKHORN = True
PARITY = False  # True only for pipeline checks; never in a submission
t_start = time.time()


def find(pattern, must=""):
    hits = [h for h in sorted(glob.glob(f"{ROOT}/**/{pattern}", recursive=True)) if must in h]
    if not hits:
        raise FileNotFoundError(f"{pattern} ({must}) not found under {ROOT}")
    return hits[0]


def P(*x):
    print(f"[{time.time() - t_start:6.0f}s]", *x, flush=True)


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def sinkhorn(lp, iters=100):
    p = np.exp(lp - lp.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    for _ in range(iters):
        p /= p.sum(0, keepdims=True)
        p /= p.sum(1, keepdims=True)
    return np.log(p + 1e-12)


# ---------------------------------------------------------------- data
train_csv = find("train.csv", "name-that-llm")
d = os.path.dirname(train_csv)
train = pd.read_csv(train_csv)
test = pd.read_csv(f"{d}/test.csv")
train_text = pd.Series(train.text).fillna("").astype(str).values
test_text = pd.Series(test.text).fillna("").astype(str).values
classes = sorted(train.model.unique())
if PARITY:  # pipeline check: score 900 train rows outside the lockbox fold with the externally trained members only
    from sklearn.model_selection import GroupKFold
    lock = list(GroupKFold(n_splits=5).split(train, train.model, train.prompt_id))[4][1]
    rest = train.drop(index=train.index[lock])
    test = rest.groupby("model").sample(n=50, random_state=0).reset_index(drop=True)
    test_text = test.text.fillna("").astype(str).values
    WEIGHTS = {k: v for k, v in WEIGHTS.items() if k != "svm"}  # the SVM is fit on these very rows
P("train", train.shape, "test", test.shape)
members = {}

# ---------------------------------------------------------------- Qwen members
qwen_names = [k.split(":", 1)[1] for k in WEIGHTS if k.startswith("qwen:")]
if qwen_names:
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
                        find("bitsandbytes-*.whl")], check=True)
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig

    MODEL = os.path.dirname(find("config.json", "4b-base"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    class MultiScale(torch.nn.Module):  # identical to finetune/train.py so the saved names line up
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

    @torch.no_grad()
    def qwen_scores(prefix):  # files <prefix>__weights.pt, __args.json, __classes.json (Kaggle flattens folders)
        args = json.load(open(f"{prefix}__args.json"))
        assert json.load(open(f"{prefix}__classes.json")) == classes, "class order differs"
        cfg = AutoConfig.from_pretrained(MODEL)
        cfg.num_hidden_layers = args["layers"] or cfg.num_hidden_layers
        if getattr(cfg, "layer_types", None):
            cfg.layer_types = cfg.layer_types[:cfg.num_hidden_layers]
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.float16)  # T4 has no native bf16
        base = AutoModel.from_pretrained(MODEL, config=cfg, quantization_config=bnb, dtype=torch.float16,
                                         device_map={"": 0})
        base.norm = torch.nn.Identity()
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)  # same fp32 casts as training
        base = get_peft_model(base, LoraConfig(r=args["lora_r"], lora_alpha=2 * args["lora_r"], lora_dropout=0.0,
                                               target_modules=LORA_TARGETS))
        depths = [int(x) for x in args["depths"].split(",")]
        model = MultiScale(base, depths, cfg.hidden_size, len(classes)).cuda().eval()
        sd = torch.load(f"{prefix}__weights.pt", map_location="cpu", weights_only=False)
        res = model.load_state_dict(sd, strict=False)
        assert not res.unexpected_keys, res.unexpected_keys[:5]
        assert not [k for k in res.missing_keys if "lora_" in k or k.startswith(("norms.", "head."))], "weights missing"
        ids = tok(list(test_text), truncation=True, max_length=args.get("eval_maxlen", 384),
                  add_special_tokens=False)["input_ids"]
        order = np.argsort([len(x) for x in ids], kind="stable")
        out = np.zeros((len(ids), len(classes)), dtype=np.float32)
        for s in range(0, len(order), 16):
            b = order[s:s + 16]
            L = max(1, max(len(ids[i]) for i in b))
            x = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
            m = torch.zeros((len(b), L), dtype=torch.long)
            for r, i in enumerate(b):
                x[r, :len(ids[i])] = torch.tensor(ids[i], dtype=torch.long)
                m[r, :len(ids[i])] = 1
            out[b] = model(x.cuda(), m.cuda()).float().cpu().numpy()
        del model, base
        torch.cuda.empty_cache()
        assert np.isfinite(out).all(), "non-finite Qwen logits"
        return out

    for name in qwen_names:
        members[f"qwen:{name}"] = qwen_scores(find(f"{name}__weights.pt")[: -len("__weights.pt")])
        P(f"qwen {name} done")

# ---------------------------------------------------------------- TF-IDF SVM (fit here on train.csv)
if "svm" in WEIGHTS:
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    def style_stats(texts):
        rows = []
        for t in texts:
            words = t.split()
            sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
            n = max(len(t), 1)
            rows.append([np.log(max(len(t), 1)), np.log(max(len(words), 1)),
                         np.mean([len(w) for w in words]) if words else 0, len(words) / max(len(sents), 1),
                         t.count(",") / n, t.count("—") / n, t.count("–") / n, t.count("-") / n,
                         t.count(";") / n, t.count(":") / n, t.count("(") / n, t.count('"') / n,
                         t.count("“") / n, t.count("’") / n, t.count("'") / n, t.count("\n") / n,
                         t.count("*") / n, sum(c.isupper() for c in t) / n,
                         sum(c.isdigit() for c in t) / n, sum(ord(c) > 127 for c in t) / n])
        return np.array(rows)

    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True, min_df=2, lowercase=False,
                           max_features=300_000)
    scaler = StandardScaler().fit(style_stats(train_text))

    def featurize(texts, fit=False):
        w = word.fit_transform(texts) if fit else word.transform(texts)
        c = char.fit_transform(texts) if fit else char.transform(texts)
        return sp.hstack([w, c, sp.csr_matrix(scaler.transform(style_stats(texts)) * 0.3)]).tocsr()

    svm = LinearSVC(C=1.0, max_iter=2000).fit(featurize(train_text, fit=True), train.model.values)
    assert list(svm.classes_) == classes
    members["svm"] = svm.decision_function(featurize(test_text))
    P("svm done")

# ---------------------------------------------------------------- LightGBM on style features
if "lgbm" in WEIGHTS:
    import lightgbm as lgb
    sys.path.insert(0, os.path.dirname(find("style_features.py")))
    from style_features import NAMES, extract  # noqa: E402

    booster = lgb.Booster(model_file=find("lgbm_ext.txt"))
    assert booster.num_feature() == len(NAMES)
    Xs = pd.DataFrame(np.asarray([extract(t) for t in test_text], dtype=np.float32), columns=NAMES)
    members["lgbm"] = np.log(np.clip(booster.predict(Xs, num_iteration=booster.best_iteration or None), 1e-9, 1))
    P("lightgbm done")

# ---------------------------------------------------------------- blend + write
blend = sum(w * log_softmax(T * members[k]) for k, (w, T) in WEIGHTS.items())
# Prompt centering: subtract CENTER_W x the average log-probabilities of the passage's prompt cluster (topic cancels,
# style remains), then Sinkhorn. OOF folds 0-3: full prompt groups 0.594 -> 0.643, random partial groups
# 0.586 -> 0.594 (features/centering.py and its follow-up check). Clusters of one passage are left untouched.
CENTER_W = 0.5
if CENTER_W and len(test) >= 10 * len(classes):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    Xg = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, max_df=0.05).fit_transform(test_text)
    Zg = normalize(TruncatedSVD(min(256, Xg.shape[1] - 1, Xg.shape[0] - 1), random_state=0).fit_transform(Xg))
    groups = AgglomerativeClustering(n_clusters=max(1, round(len(test) / len(classes))), metric="cosine",
                                     linkage="average").fit_predict(Zg)
    lb = log_softmax(blend)
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= 2:
            blend[m] = lb[m] - CENTER_W * lb[m].mean(0)
    P("prompt centering:", len(set(groups)), "clusters, sizes", pd.Series(groups).value_counts().describe().round(1).to_dict())
if PARITY:
    from sklearn.metrics import balanced_accuracy_score as bas
    y = test.model.values
    for k, z in list(members.items()) + [("blend", blend)]:
        print(f"PARITY {k}: raw {bas(y, np.array(classes)[z.argmax(1)]):.4f} "
              f"balanced {bas(y, np.array(classes)[sinkhorn(log_softmax(z)).argmax(1)]):.4f}", flush=True)
if SINKHORN and len(test) >= 10 * len(classes):  # only meaningful on a reasonably large, balanced test set
    blend = sinkhorn(blend)
pred = np.array(classes)[blend.argmax(1)]
sub = pd.DataFrame({"id": test.id.astype(str).values, "model": pred})
assert len(sub) == len(test)
out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("NTL_OUT", ".")
sub.to_parquet(f"{out_dir}/submission.parquet", index=False)
P(sub.model.value_counts().to_string(), "\nwritten", len(sub), "rows")
