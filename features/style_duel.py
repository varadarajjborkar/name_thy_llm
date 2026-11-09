"""Sibling re-ranking with INDEPENDENT evidence: our 316 hand-made style features (v40-style idea, our own version).

Same protocol as features/duel.py: one classifier per sibling group, re-ranks only inside the group the blend
picked, keeps that group's total probability, scored out of fold on folds 0-3 (never fold 4). The difference is the
evidence: punctuation, sentence shape and typography counts that the Qwen models never see directly.
Training rows per fold k: sibling pool passages (external, never host) + host rows of the group outside fold k.
Run: .venv/Scripts/python.exe features/style_duel.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "features"))
sys.path.insert(0, str(ROOT / "finetune"))
from prep_data import typography_mask  # noqa: E402
from style_features import extract  # noqa: E402

P = ROOT / "data/processed"
GROUPS = {
    "glm": ["glm-5.3", "glm-5.3-flash"],
    "gemini_flash": ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"],
    "gpt56": ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"],
    "claude": ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5"],
}
HOST_WEIGHT = 5.0
train = pd.read_csv(ROOT / "data/train.csv")
train["id"] = train.id.astype(str)
classes = sorted(train.model.unique())
cid = {c: i for i, c in enumerate(classes)}
fold_of = {}
for k, (_, va) in enumerate(GroupKFold(5).split(train, train.model, train.prompt_id)):
    fold_of.update({i: k for i in train.id.iloc[va]})
bl = np.load(P / "blend_oof.npz", allow_pickle=True)
ids = [str(i) for i in bl["ids"]]
lp, y = bl["lp"], bl["y"]
host = train.set_index("id").loc[ids]
hfold = np.array([fold_of[i] for i in ids])
hS = np.asarray([extract(t) for t in host.text.fillna("").astype(str)], dtype=np.float32)

sib = sorted({m for g in GROUPS.values() for m in g})
pool = pd.concat([pd.read_parquet(P / f, columns=["text", "model"]) for f in ("reserve_typo.parquet", "pool_typo_60k.parquet")])
pool = pool[pool.model.isin(sib) & typography_mask(pool) & ~pool.text.isin(set(train.text))].drop_duplicates("text")
pool = pd.concat([g.sample(n=min(len(g), 2000), random_state=5) for _, g in pool.groupby("model")])
pS = np.asarray([extract(t) for t in pool.text.astype(str)], dtype=np.float32)
pS, hS = np.nan_to_num(pS), np.nan_to_num(hS)
print(f"host {len(y)} | pool {len(pool)} | blend raw {bacc(y, lp.argmax(1)):.4f}")

spec = {g: np.full((len(y), len(m)), np.nan) for g, m in GROUPS.items()}
for k in range(4):
    for g, members in GROUPS.items():
        pm = pool.model.isin(members).to_numpy()
        hm = host.model.isin(members).to_numpy() & (hfold != k)
        X = np.vstack([pS[pm], hS[hm]])
        yy = np.concatenate([pool.model[pm].to_numpy(), host.model[hm].to_numpy()])
        w = np.concatenate([np.ones(pm.sum()), np.full(hm.sum(), HOST_WEIGHT)])
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(sc.transform(X), yy, sample_weight=w)
        order = [list(clf.classes_).index(m) for m in members]
        va = hfold == k
        spec[g][va] = clf.predict_proba(sc.transform(hS[va]))[:, order]

parts = []
for g, members in GROUPS.items():
    idx = [cid[m] for m in members]
    rows = np.isin(y, idx)
    parts.append(f"{g} blend {np.mean(np.array(idx)[lp[rows][:, idx].argmax(1)] == y[rows]):.3f} "
                 f"style {np.mean(np.array(idx)[spec[g][rows].argmax(1)] == y[rows]):.3f}")
print("within-group accuracy: " + " | ".join(parts))
top = lp.argmax(1)
for lam in (0.1, 0.25, 0.5, 1.0):
    new = lp.copy()
    for g, members in GROUPS.items():
        idx = np.array([cid[m] for m in members])
        rows = np.isin(top, idx)
        sub = new[np.ix_(rows, idx)]
        mass = np.log(np.exp(sub).sum(1, keepdims=True))
        sub = sub + lam * np.log(np.clip(spec[g][rows], 1e-6, 1))
        new[np.ix_(rows, idx)] = sub - np.log(np.exp(sub).sum(1, keepdims=True)) + mass
    print(f"lam {lam}: balanced accuracy {bacc(y, new.argmax(1)):.4f} (blend {bacc(y, lp.argmax(1)):.4f})")
