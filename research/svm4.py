"""Component 3: four-view n-gram SVM, trained on host rows (and optionally pool rows), as a drop-in SVM member.

Views: word 1-2 grams | char 2-5 grams | shape (letters -> a/A, digits -> 0: punctuation, case and spacing habits) |
topic-masked (non-function words -> w, then word 1-3 grams: phrasing structure without topic).
Out of fold on folds 0-3: for fold k, fit on host rows outside fold k (+ pool rows), score fold k. The blend
(0.8 stage-2 families + 0.1 SVM + 0.1 LightGBM) is re-scored with the new SVM in place of the old one.
Run: .venv/Scripts/python.exe research/svm4.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from assign import log_softmax, sinkhorn  # noqa: E402

FUNC = set("""a about above after again against all am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her here hers him
his how i if in into is it its itself just me might more most must my no nor not now of off on once only or other our
out over own same she should so some such than that the their them then there these they this those through to too
under until up very was we were what when where which while who whom why will with would you your however therefore
moreover although perhaps also even still yet thus rather often""".split())
TOK = re.compile(r"[A-Za-z]+|[^A-Za-z\s]")


def shape(t):
    return re.sub(r"\d", "0", re.sub(r"[a-z]", "a", re.sub(r"[A-Z]", "A", t)))


def masked(t):
    return " ".join(w if (w.lower() in FUNC or not w.isalpha()) else "w" for w in TOK.findall(t))


def views():
    return [("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=300_000), None),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True, min_df=2,
                                     lowercase=False, max_features=300_000), None),
            ("shape", TfidfVectorizer(analyzer="char", ngram_range=(1, 5), sublinear_tf=True, min_df=2,
                                      lowercase=False, max_features=100_000), shape),
            ("masked", TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=2, lowercase=False,
                                       token_pattern=r"\S+", max_features=200_000), masked)]


def fit_score(tr_text, tr_y, tr_w, te_text):
    mats_tr, mats_te = [], []
    for _, vec, f in views():
        a = [f(t) for t in tr_text] if f else list(tr_text)
        b = [f(t) for t in te_text] if f else list(te_text)
        mats_tr.append(vec.fit_transform(a))
        mats_te.append(vec.transform(b))
    clf = LinearSVC(C=0.5, max_iter=3000).fit(sp.hstack(mats_tr).tocsr(), tr_y, sample_weight=tr_w)
    return clf.decision_function(sp.hstack(mats_te).tocsr()), list(clf.classes_)


d = np.load(ROOT / "data/processed/blend_parts.npz", allow_pickle=True)
ids, y = [str(i) for i in d["ids"]], d["y"]
train = pd.read_csv(ROOT / "data/train.csv")
train["id"] = train.id.astype(str)
classes = sorted(train.model.unique())
fold_of = {i: k for k, (_, va) in enumerate(GroupKFold(5).split(train, train.model, train.prompt_id))
           for i in train.id.iloc[va]}
text_all = train.text.fillna("").astype(str).to_numpy()
fold_all = train.id.map(fold_of).to_numpy()
pos = pd.Series(np.arange(len(train)), index=train.id).loc[ids].to_numpy()
pool = pd.read_parquet(ROOT / "data/processed/pool_typo_60k.parquet", columns=["text", "model"])
pool = pd.concat([g.sample(n=2000, random_state=6) for _, g in pool.groupby("model")])


def blend_with(svm_scores):
    b = log_softmax(0.8 * d["s2mix"] + 0.1 * log_softmax(3 * svm_scores) + 0.1 * d["lgb"])
    return bacc(y, b.argmax(1)), bacc(y, (0.55 * b + 0.45 * sinkhorn(b)).argmax(1))


print(f"old SVM member: alone {bacc(y, d['svm'].argmax(1)):.4f} | blend argmax / partial-sinkhorn "
      f"{blend_with(d['svm'] / 3)[0]:.4f} / {blend_with(d['svm'] / 3)[1]:.4f}", flush=True)
for name, use_pool, pool_w in (("4-view host only", False, 0), ("4-view host + pool (w 0.3)", True, 0.3)):
    oof = np.zeros((len(y), len(classes)))
    for k in range(4):
        tr = fold_all != k
        txt, yy, w = list(text_all[tr]), list(train.model.to_numpy()[tr]), [1.0] * int(tr.sum())
        if use_pool:
            txt += list(pool.text)
            yy += list(pool.model)
            w += [pool_w] * len(pool)
        va = np.flatnonzero(np.array([fold_of[i] for i in ids]) == k)
        s, cl = fit_score(txt, np.array(yy), np.array(w), text_all[pos[va]])
        assert cl == classes
        oof[va] = s
        print(f"  {name}: fold {k} done", flush=True)
    a, b = blend_with(oof)
    print(f"{name}: alone {bacc(y, oof.argmax(1)):.4f} | blend argmax / partial-sinkhorn {a:.4f} / {b:.4f}", flush=True)
    np.save(ROOT / f"data/processed/svm4_oof_{'pool' if use_pool else 'host'}.npy", oof)
