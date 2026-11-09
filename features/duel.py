"""Duel specialists: challengers that fight the blend's pick inside a sibling group, scored honestly on folds 0-3.

For each sibling group (GLM pair, Gemini flash trio, GPT-5.6 trio, Claude family) a small classifier separates only
those models, from frozen stage-1 Qwen features squeezed to a few dimensions (PCA: the dimension knob).
When the blend's top pick is in a group, the specialist re-scores the group's members:
    score(c) = log p_blend(c) + lam * log p_specialist(c)     for c in the group, other classes untouched.
Training rows per fold k: sibling pool passages (unseen by the feature model) + host rows of the group outside fold k
(never fold 4, the lockbox). Everything printed is out-of-fold.
Run: .venv/Scripts/python.exe features/duel.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "features"))
P = ROOT / "data/processed"
RUN = "run3"
GROUPS = {
    "glm": ["glm-5.3", "glm-5.3-flash"],
    "gemini_flash": ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"],
    "gpt56": ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"],
    "claude": ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5"],
}
HOST_WEIGHT = 5.0  # host rows are the target distribution; pool rows are many but shifted

train = pd.read_csv(ROOT / "data/train.csv")
classes = sorted(train.model.unique())
cid = {c: i for i, c in enumerate(classes)}
fold_of = {}
for k, (_, va) in enumerate(GroupKFold(5).split(train, train.model, train.prompt_id)):
    fold_of.update({str(i): k for i in train.id.iloc[va]})

hX = np.load(P / f"feats_{RUN}_host.npy").astype(np.float32)
hmeta = pd.read_parquet(P / f"feats_{RUN}_host.parquet")
pX = np.load(P / f"feats_{RUN}_pool.npy").astype(np.float32)
pmeta = pd.read_parquet(P / f"feats_{RUN}_pool.parquet")
bl = np.load(P / "blend_oof.npz", allow_pickle=True)
pos = pd.Series(np.arange(len(hmeta)), index=hmeta.id.astype(str)).loc[[str(i) for i in bl["ids"]]].to_numpy()
hX, hmeta = hX[pos], hmeta.iloc[pos].reset_index(drop=True)
lp, y = bl["lp"], bl["y"]
hfold = hmeta.id.astype(str).map(fold_of).to_numpy()
assert (hmeta.model.map(cid).to_numpy() == y).all() and not (hfold == 4).any()
print(f"host rows {len(y)} | pool rows {len(pmeta)} | blend raw {bacc(y, lp.argmax(1)):.4f}")


def specialist_probs(dims, k, members):
    """Train on pool + host rows of the group outside fold k; return probs for every fold-k host row."""
    pm = pmeta.model.isin(members).to_numpy()
    hm = hmeta.model.isin(members).to_numpy() & (hfold != k)
    X = np.vstack([pX[pm], hX[hm]])
    yy = np.concatenate([pmeta.model[pm].to_numpy(), hmeta.model[hm].to_numpy()])
    w = np.concatenate([np.ones(pm.sum()), np.full(hm.sum(), HOST_WEIGHT)])
    sc = StandardScaler().fit(X)
    pca = PCA(n_components=dims, random_state=0).fit(sc.transform(X))
    Z = pca.transform(sc.transform(X))
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Z, yy, sample_weight=w)
    order = [list(clf.classes_).index(m) for m in members]
    va = hfold == k
    return va, clf.predict_proba(pca.transform(sc.transform(hX[va])))[:, order]


for dims in (32, 128, 512):
    spec = {g: np.full((len(y), len(m)), np.nan) for g, m in GROUPS.items()}
    for k in range(4):
        for g, members in GROUPS.items():
            va, pr = specialist_probs(dims, k, members)
            spec[g][va] = pr
    # how well each specialist separates its siblings on host rows truly from the group, vs the blend
    parts = []
    for g, members in GROUPS.items():
        idx = [cid[m] for m in members]
        rows = np.isin(y, idx)
        b_in = np.array(idx)[lp[rows][:, idx].argmax(1)]
        s_in = np.array(idx)[spec[g][rows].argmax(1)]
        parts.append(f"{g} blend {np.mean(b_in == y[rows]):.3f} spec {np.mean(s_in == y[rows]):.3f}")
    print(f"\ndims {dims} | within-group accuracy (rows truly in the group): " + " | ".join(parts))
    for lam in (0.5, 1.0, 2.0):
        new = lp.copy()
        top = lp.argmax(1)
        for g, members in GROUPS.items():
            idx = np.array([cid[m] for m in members])
            rows = np.isin(top, idx)
            sub = new[np.ix_(rows, idx)]
            mass = np.log(np.exp(sub).sum(1, keepdims=True))
            sub = sub + lam * np.log(np.clip(spec[g][rows], 1e-6, 1))
            sub = sub - np.log(np.exp(sub).sum(1, keepdims=True)) + mass  # keep the group's total probability
            new[np.ix_(rows, idx)] = sub
        print(f"  lam {lam}: balanced accuracy {bacc(y, new.argmax(1)):.4f} (blend {bacc(y, lp.argmax(1)):.4f})")
