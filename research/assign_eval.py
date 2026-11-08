"""Does the 'one passage per model per prompt' structure help? Tested on out-of-fold predictions of train.csv.

train.csv: 150 prompts, each answered once by each of the 18 models. If the hidden test is built the same way,
passages can be grouped by topic and each group assigned one-model-each (Hungarian), instead of argmax per row.
  1. oracle groups  : true prompt_id (upper bound of the idea)
  2. found groups   : prompt_id hidden, groups recovered by clustering on content words (what the notebook can do)
Inputs: SVM OOF scores (cached) + LightGBM external-model probabilities, same blend as the submitted notebook.
Run: .venv/Scripts/python.exe research/assign_eval.py
"""
import ctypes
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception:
    pass
from assign import assign_groups, find_groups, log_softmax, sinkhorn  # noqa: E402

df = pd.read_csv(ROOT / "data/train.csv")
text = df.text.fillna("").astype(str).values
classes = sorted(df.model.unique())
y = df.model.map({c: i for i, c in enumerate(classes)}).to_numpy()

cache = ROOT / "data/processed/svm_oof.npy"
if cache.exists():
    oof = np.load(cache)
else:
    src = (ROOT / "features/blend_eval.py").read_text(encoding="utf-8").split("\ndf = pd.read_csv")[0]
    ns = {"__file__": str(ROOT / "features/blend_eval.py")}
    exec(compile(src, "blend_eval_defs", "exec"), ns)  # reuse the exact SVM definition, skip its main block
    oof = np.zeros((len(df), len(classes)))
    for k, (tr, va) in enumerate(GroupKFold(5).split(df, y, df.prompt_id)):
        s, cl = ns["svm_scores"](text[tr], df.model.values[tr], text[va])
        assert cl == classes
        oof[va] = s
        print(f"svm fold {k}", flush=True)
    np.save(cache, oof)
lg = np.log(np.clip(np.load(ROOT / "data/processed/lgbm_ext_val_probs.npy"), 1e-9, 1))
blend = 0.5 * log_softmax(3 * oof) + 0.5 * lg

print(f"argmax            {bacc(y, blend.argmax(1)):.4f}")
print(f"sinkhorn          {bacc(y, sinkhorn(blend).argmax(1)):.4f}")
print(f"oracle groups     {bacc(y, assign_groups(blend, df.prompt_id.to_numpy())):.4f}")
g = find_groups(text, n_per_group=len(classes))
print(f"found groups: {len(set(g))} groups, ARI vs true prompts {adjusted_rand_score(df.prompt_id, g):.3f}, "
      f"sizes {pd.Series(g).value_counts().describe()[['min', '50%', 'max']].to_dict()}")
print(f"found groups      {bacc(y, assign_groups(blend, g)):.4f}")
