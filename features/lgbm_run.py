"""LightGBM on hand-made style features.

  .venv\\Scripts\\python.exe features\\lgbm_run.py cv     # 5-fold GroupKFold by prompt_id on train.csv only
  .venv\\Scripts\\python.exe features\\lgbm_run.py ext    # train on the 100k external sample, score on train.csv

Polite by design: lower process priority and few threads, so it never slows the GPU training run.
Saves probabilities for later stacking in data/processed/ (cv: out-of-fold; ext: predictions on train.csv).
"""
import argparse
import ctypes
import multiprocessing as mp
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style_features import NAMES, extract  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
THREADS = 6  # of 24; the trainer needs the rest


def lower_priority():
    try:  # BELOW_NORMAL_PRIORITY_CLASS
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:
        pass


def code_hash():
    """Cache key: changes whenever the feature code or its column list changes."""
    import hashlib
    src = (Path(__file__).resolve().parent / "style_features.py").read_bytes()
    return hashlib.sha1(src + "|".join(NAMES).encode()).hexdigest()[:8]


def feats(texts, cache):
    texts = list(texts)
    cache = cache.with_name(f"{cache.stem}_{code_hash()}{cache.suffix}")
    if cache.exists():
        df = pd.read_parquet(cache)
        if len(df) == len(texts) and list(df.columns) == NAMES:
            return df
        print(f"  cache {cache.name} does not match the data, rebuilding", flush=True)
    t = time.time()
    with mp.Pool(THREADS, initializer=lower_priority) as pool:
        rows = pool.map(extract, texts, chunksize=500)
    df = pd.DataFrame(np.asarray(rows, dtype=np.float32), columns=NAMES)
    df.to_parquet(cache.with_suffix(".tmp"))
    cache.with_suffix(".tmp").replace(cache)  # atomic: a killed run never leaves a half-written cache
    print(f"  extracted {len(df):,} x {len(NAMES)} features in {time.time() - t:.0f}s -> {cache.name}", flush=True)
    return df


def params(n_classes, **kw):
    p = dict(objective="multiclass", num_class=n_classes, learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
             feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=THREADS,
             verbose=-1, seed=42)
    p.update(kw)
    return p


def top_features(booster, k=20):
    imp = pd.Series(booster.feature_importance("gain"), index=NAMES).sort_values(ascending=False)
    print("top features by gain:", ", ".join(f"{n} ({v / imp.sum():.1%})" for n, v in imp.head(k).items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cv", "ext"])
    ap.add_argument("--rounds", type=int, default=300)
    a = ap.parse_args()
    lower_priority()
    PROC.mkdir(parents=True, exist_ok=True)

    comp = pd.read_csv(ROOT / "data" / "train.csv")
    classes = sorted(comp.model.unique())
    y = comp.model.map({c: i for i, c in enumerate(classes)}).to_numpy()
    Xc = feats(comp.text, PROC / "feat_comp.parquet")
    assert len(Xc) == len(comp)

    if a.mode == "cv":
        oof = np.zeros((len(comp), len(classes)), dtype=np.float32)
        for k, (tr, va) in enumerate(GroupKFold(5).split(Xc, y, comp.prompt_id)):
            m = lgb.train(params(len(classes)), lgb.Dataset(Xc.iloc[tr], y[tr]), num_boost_round=a.rounds)
            oof[va] = m.predict(Xc.iloc[va])
            print(f"fold {k}: balanced acc {balanced_accuracy_score(y[va], oof[va].argmax(1)):.4f}", flush=True)
        acc = balanced_accuracy_score(y, oof.argmax(1))
        print(f"\nCV balanced accuracy (grouped by prompt): {acc:.4f}   [TF-IDF+SVM baseline 0.368, chance 0.056]")
        np.save(PROC / "lgbm_cv_oof.npy", oof)
        top_features(m)
    else:
        ext = pd.read_parquet(PROC / "train_100k.parquet", columns=["text", "model", "prompt_group"])
        Xe = feats(ext.text, PROC / "feat_ext100k.parquet")
        ye = ext.model.map({c: i for i, c in enumerate(classes)}).to_numpy()
        tr, va = next(GroupShuffleSplit(1, test_size=0.05, random_state=0).split(Xe, ye, ext.prompt_group))
        t = time.time()
        m = lgb.train(params(len(classes), learning_rate=0.08), lgb.Dataset(Xe.iloc[tr], ye[tr]), num_boost_round=1500,
                      valid_sets=[lgb.Dataset(Xe.iloc[va], ye[va])],
                      callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(100)])
        print(f"trained {m.best_iteration} rounds in {time.time() - t:.0f}s", flush=True)
        p_in = m.predict(Xe.iloc[va], num_iteration=m.best_iteration)
        print(f"held-out external (same distribution) balanced acc: {balanced_accuracy_score(ye[va], p_in.argmax(1)):.4f}")
        pc = m.predict(Xc, num_iteration=m.best_iteration)
        acc = balanced_accuracy_score(y, pc.argmax(1))
        print(f"\ntrain.csv balanced accuracy (never trained on): {acc:.4f}   [TF-IDF+SVM baseline 0.368, chance 0.056]")
        np.save(PROC / "lgbm_ext_val_probs.npy", pc.astype(np.float32))
        m.save_model(str(PROC / "lgbm_ext.txt"))
        top_features(m)


if __name__ == "__main__":
    main()
