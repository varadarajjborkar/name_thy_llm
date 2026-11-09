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
# v6: + s4all (stage 2 from run4, stage 1 on 92k pool rows). Fold 0 with centering + partial sinkhorn: old blend
# 0.639, s4 weight 0.5 (old families 0.1 each) 0.663; s4 alone on fold 0 0.617 vs 0.567 / 0.567 / 0.559.
WEIGHTS = {"qwen:s4all": (0.5, 1.0), "qwen:s2all": (0.1, 1.0), "qwen:s3all": (0.1, 1.0), "qwen:h3s2all": (0.1, 1.0),
           "svm": (0.1, 3.0), "lgbm": (0.1, 1.0),
           "atlas": (0.15, 1.0)}  # v7: on top of the v6 weights (they sum to 1), as evaluated out of fold
SINKHORN = True
SINK_ALPHA = 0.45  # LB history: full sinkhorn + centering w0.5 = 0.572 (v5)
PARITY = os.environ.get("NTL_PARITY", "0") == "1"  # pipeline check only (env switch for local tests); never in a submission
t_start = time.time()


def find(pattern, must=""):
    hits = [h for h in sorted(glob.glob(f"{ROOT}/**/{pattern}", recursive=True)) if must in h]
    if not hits:
        raise FileNotFoundError(f"{pattern} ({must}) not found under {ROOT}")
    return hits[0]


OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("NTL_OUT", ".")
SUMMARY = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": [], "checks": {}}


def gpu_mem():
    try:
        import torch
        if torch.cuda.is_available():
            return f"gpu {torch.cuda.memory_allocated() / 2**30:.1f}/{torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
    except Exception:
        pass
    return ""


def P(*x):
    print(f"[{time.time() - t_start:6.0f}s]", *x, gpu_mem(), flush=True)


def stage(name, **info):
    """Log a finished stage and rewrite run_summary.json, so even a crashed run leaves a trail in its output."""
    SUMMARY["stages"].append({"stage": name, "t": round(time.time() - t_start, 1), **info})
    P(f"STAGE {name}", " ".join(f"{k}={v}" for k, v in info.items()))
    try:
        with open(f"{OUT_DIR}/run_summary.json", "w") as f:
            json.dump(SUMMARY, f, indent=1, default=str)
    except Exception as e:
        print("summary write failed:", e, flush=True)


def _has(md, k):
    try:
        md.version(k)
        return True
    except Exception:
        return False


def check_member(k, z):
    """Shape, finiteness and how spread the member's votes are (a collapsed member votes one class)."""
    assert z.shape == (len(test), len(classes)), f"{k}: shape {z.shape}"
    assert np.isfinite(z).all(), f"{k}: non-finite scores"
    top = pd.Series(z.argmax(1)).value_counts()
    SUMMARY["checks"][k] = {"top_class_share": round(float(top.iloc[0] / len(z)), 3), "classes_voted": int(len(top))}


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
    WEIGHTS = {k: v for k, v in WEIGHTS.items() if k not in ("svm", "atlas")}  # both are fit on these very rows
P("train", train.shape, "test", test.shape)
try:
    import importlib.metadata as md
    SUMMARY["versions"] = {k: md.version(k) for k in ("torch", "transformers", "peft", "bitsandbytes", "lightgbm",
                                                        "scikit-learn") if _has(md, k)}
except Exception as e:
    SUMMARY["versions"] = str(e)
stage("data", train=len(train), test=len(test), parity=PARITY)
members = {}

# ---------------------------------------------------------------- base model, staged to local disk
# Kaggle staff (competition forum 750924): /kaggle/input reads can crawl on some GPU sessions (mmap-style model
# loading), which turned 15-minute scorings into hours; a plain sequential copy to /tmp runs at normal speed.
# The base model is loaded five times (four classifiers + ATLAS), so it is copied once and read from local disk.
def stage_model(src):
    import shutil
    import tempfile
    on_kaggle = src.startswith("/kaggle/") or os.environ.get("NTL_STAGE") == "1"
    dst = os.path.join(tempfile.gettempdir(), "qwen3-4b-base")
    if not on_kaggle:
        return src
    try:
        need = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(src) for f in fs)
        if shutil.disk_usage(tempfile.gettempdir()).free < need + 4 * 2**30:
            stage("model staging skipped (disk)", need_gb=round(need / 2**30, 1))
            return src
        t0 = time.time()
        shutil.copytree(src, dst, dirs_exist_ok=True)
        got = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(dst) for f in fs)
        if got != need:
            raise OSError(f"copy incomplete: {got} of {need} bytes")
        stage("model staged to local disk", gb=round(need / 2**30, 1), seconds=round(time.time() - t0, 1))
        return dst
    except Exception as e:  # any copy problem: remove the partial copy and read from the input mount as before
        shutil.rmtree(dst, ignore_errors=True)
        stage("model staging failed, reading from input", error=f"{type(e).__name__}: {e}")
        return src


BASE_DIR = None
if any(k.startswith("qwen:") for k in WEIGHTS) or "atlas" in WEIGHTS or PARITY:
    BASE_DIR = stage_model(os.path.dirname(find("config.json", "4b-base")))

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

    MODEL = BASE_DIR
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
        import gc
        gc.collect()  # peft / bitsandbytes modules can hold reference cycles; free before the next member loads
        torch.cuda.empty_cache()
        assert np.isfinite(out).all(), "non-finite Qwen logits"
        return out

    for name in qwen_names:
        members[f"qwen:{name}"] = qwen_scores(find(f"{name}__weights.pt")[: -len("__weights.pt")])
        check_member(f"qwen:{name}", members[f"qwen:{name}"])
        stage(f"qwen {name}", rows=len(test))

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
    check_member("svm", members["svm"])
    stage("svm")

# ---------------------------------------------------------------- LightGBM on style features
if "lgbm" in WEIGHTS:
    import lightgbm as lgb
    sys.path.insert(0, os.path.dirname(find("style_features.py")))
    from style_features import NAMES, extract  # noqa: E402

    booster = lgb.Booster(model_file=find("lgbm_ext.txt"))
    assert booster.num_feature() == len(NAMES)
    Xs = pd.DataFrame(np.asarray([extract(t) for t in test_text], dtype=np.float32), columns=NAMES)
    members["lgbm"] = np.log(np.clip(booster.predict(Xs, num_iteration=booster.best_iteration or None), 1e-9, 1))
    check_member("lgbm", members["lgbm"])
    stage("lightgbm")

# ---------------------------------------------------------------- ATLAS: how each passage looks to the plain base LM
# Per-token log-prob / entropy / top-1 statistics under Qwen3-4B base (4-bit, fp16), then 5 class-balanced LightGBM
# seeds trained on all train.csv rows + 2,950 pool rows (features/atlas_fit.py). OOF folds 0-3, seed-averaged:
# + 0.15 x ATLAS on the v6 blend = +1.08 (every fold up). The feature code is the same module used for training.
ATLAS_PROBE = PARITY  # parity: compute and save ATLAS features (in-sample there, so they stay out of the blend)
if "atlas" in WEIGHTS or ATLAS_PROBE:
    try:  # never lose a submission to this member: on any failure the blend falls back to exactly v6
        import lightgbm as lgb
        import torch
        sys.path.insert(0, os.path.dirname(find("atlas_feats.py")))
        from atlas_feats import load_lm, score  # noqa: E402

        meta = json.load(open(find("atlas_meta.json")))
        assert meta["classes"] == classes, "ATLAS class order differs"
        tok_a, lm_a = load_lm(BASE_DIR, torch.float16)
        Fa = score(list(test_text), tok_a, lm_a, 4, meta["maxlen"], log=P)
        del lm_a
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        Xa = Fa[meta["cols"]].to_numpy()
        bad = ~np.isfinite(Xa).all(1)
        if len(Xa) >= 100 and "mean" in meta:  # drift guard: features unlike the training ones -> leave ATLAS out
            z = np.abs((np.nanmean(np.where(np.isfinite(Xa), Xa, np.nan), 0) - np.array(meta["mean"]))
                       / np.array(meta["sd"]))
            SUMMARY["atlas_drift"] = {"mean_abs_z": round(float(np.nanmean(z)), 3),
                                      "max_abs_z": round(float(np.nanmax(z)), 3) if np.isfinite(z).any() else None}
            # worst legitimate case measured on train: a test made of ONE model only, mean |z| 0.66, max 0.94;
            # a broken library shifts by several sd. A column with no finite value at all also fails (fail closed).
            if not np.isfinite(z).all() or z.mean() > 1.0 or z.max() > 2.0:
                raise RuntimeError(f"ATLAS features drifted from training (mean |z| {z.mean():.2f}, max {z.max():.2f})")
        pa = np.mean([lgb.Booster(model_file=find(f"atlas_lgb_s{s}.txt")).predict(np.nan_to_num(Xa))
                      for s in range(meta["seeds"])], 0)
        pa[bad] = 1.0 / len(classes)  # a row with non-finite features gets a flat (no-opinion) ATLAS vote
        members["atlas"] = np.log(np.clip(pa, 1e-9, 1))
        if ATLAS_PROBE:
            Fa.assign(id=test.id.astype(str).values).to_parquet(f"{OUT_DIR}/atlas_parity_features.parquet")
        check_member("atlas", members["atlas"])
        stage("atlas", nonfinite_rows=int(bad.sum()), in_blend="atlas" in WEIGHTS)
    except Exception as e:
        WEIGHTS.pop("atlas", None)
        members.pop("atlas", None)
        SUMMARY["atlas_error"] = f"{type(e).__name__}: {e}"
        stage("atlas FAILED (blend falls back to v6)", error=f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- blend + write
missing = [k for k in WEIGHTS if k not in members]
assert not missing, f"weighted members missing: {missing}"
blend = sum(w * log_softmax(T * members[k]) for k, (w, T) in WEIGHTS.items())
SUMMARY["weights"] = {k: v[0] for k, v in WEIGHTS.items()}
# Prompt centering: subtract CENTER_W x the average log-probabilities of the passage's prompt cluster (topic cancels,
# style remains), then Sinkhorn. OOF folds 0-3: full prompt groups 0.594 -> 0.643, random partial groups
# 0.586 -> 0.594 (features/centering.py and its follow-up check). Clusters of one passage are left untouched.
CENTER_W = 0.5


def prompt_centering(z):
    """Cluster test passages by topic (TF-IDF + SVD + average-link cosine) and subtract CENTER_W x each cluster's mean
    log-probabilities. Rows whose TF-IDF vector is empty (only stop words or very common words) stay in their own
    cluster and are left untouched."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    Xg = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, max_df=0.05).fit_transform(test_text)
    ok = np.asarray(Xg.getnnz(axis=1) > 0)  # cosine affinity is undefined for all-zero rows
    groups = np.arange(len(test)) + len(test)  # unclustered rows: singleton labels that never collide
    if ok.sum() >= 2 * len(classes):
        Zg = normalize(TruncatedSVD(min(256, Xg.shape[1] - 1, int(ok.sum()) - 1), random_state=0)
                       .fit_transform(Xg[ok]))
        groups[ok] = AgglomerativeClustering(n_clusters=max(1, round(ok.sum() / len(classes))), metric="cosine",
                                             linkage="average").fit_predict(Zg)
    lb = log_softmax(z)
    out = z.copy()
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= 2:
            out[m] = lb[m] - CENTER_W * lb[m].mean(0)
    sizes = pd.Series(groups).value_counts()
    stage("prompt centering", clusters=len(sizes), multi_clusters=int((sizes >= 2).sum()), largest=int(sizes.max()),
          empty_rows=int((~np.asarray(Xg.getnnz(axis=1) > 0)).sum()))
    return out


if CENTER_W and len(test) >= 10 * len(classes):
    try:  # never lose the run here: on any clustering problem the blend is used uncentred
        blend = prompt_centering(blend)
    except Exception as e:
        SUMMARY["centering_error"] = f"{type(e).__name__}: {e}"
        stage("prompt centering FAILED (blend used uncentred)", error=f"{type(e).__name__}: {e}")
if PARITY:
    from sklearn.metrics import balanced_accuracy_score as bas
    y = test.model.values
    for k, z in list(members.items()) + [("blend", blend)]:
        r, b = bas(y, np.array(classes)[z.argmax(1)]), bas(y, np.array(classes)[sinkhorn(log_softmax(z)).argmax(1)])
        SUMMARY.setdefault("parity", {})[k] = {"raw": round(r, 4), "balanced": round(b, 4)}
        print(f"PARITY {k}{' (in-sample, not in blend)' if k == 'atlas' else ''}: raw {r:.4f} balanced {b:.4f}",
              flush=True)
if SINKHORN and len(test) >= 10 * len(classes):  # only meaningful on a reasonably large, balanced test set
    # partial: move SINK_ALPHA of the way towards equal shares (OOF: 0.45 is near-best when the test is balanced
    # and costs only 0.2 when it is not; full balancing cost 1.2 on unbalanced resamples). research/partial_sinkhorn.py
    lb = log_softmax(blend)
    blend = (1 - SINK_ALPHA) * lb + SINK_ALPHA * sinkhorn(lb)
pred = np.array(classes)[blend.argmax(1)]
sub = pd.DataFrame({"id": test.id.astype(str).values, "model": pred})
assert len(sub) == len(test) and sub.id.is_unique and sub.model.isin(classes).all()
sub.to_parquet(f"{OUT_DIR}/submission.parquet", index=False)
vc = sub.model.value_counts()
SUMMARY["prediction_counts"] = vc.to_dict()
stage("written", rows=len(sub), classes_predicted=len(vc), top_share=round(float(vc.iloc[0] / len(sub)), 3))
P(sub.model.value_counts().to_string(), "\nwritten", len(sub), "rows")
