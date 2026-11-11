# ---------------------------------------------------------------- kernel main (tree/kernel_main.py)
# Runs after style_features.py, ntl_tree.py and CONFIG (pasted above by tools/build_tree_kernel.py).
# CPU only, no internet, no external data: trains the trees on train.csv and predicts test.csv.
# Local check: NTL_ROOT=<folder holding train.csv/test.csv> NTL_OUT=<out dir> NTL_PARITY_FOLD=k holds out train fold k
# (GroupKFold(5) by prompt_id) as a labelled test; the score must match research/tree_cv.py's fold k exactly.
import glob
import json
import os
import time

ROOT_IN = os.environ.get("NTL_ROOT", "/kaggle/input")
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("NTL_OUT", ".")
PARITY_FOLD = int(os.environ.get("NTL_PARITY_FOLD", "-1"))  # local pipeline check only; never in a submission
N_JOBS = int(os.environ.get("NTL_JOBS", "4"))
os.makedirs(OUT_DIR, exist_ok=True)
t_start = time.time()
SUMMARY = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "config": CONFIG, "stages": []}


def stage(name, **info):
    """Log a finished stage and rewrite run_summary.json, so even a crashed run leaves a trail."""
    SUMMARY["stages"].append({"stage": name, "t": round(time.time() - t_start, 1), **info})
    print(f"[{time.time() - t_start:6.0f}s] STAGE {name}", " ".join(f"{k}={v}" for k, v in info.items()), flush=True)
    with open(f"{OUT_DIR}/run_summary.json", "w") as f:
        json.dump(SUMMARY, f, indent=1, default=str)


hits = [h for h in sorted(glob.glob(f"{ROOT_IN}/**/train.csv", recursive=True)) if "name-that-llm" in h
        or os.environ.get("NTL_ROOT")]
assert hits, f"train.csv not found under {ROOT_IN}"
d = os.path.dirname(hits[0])
train = pd.read_csv(f"{d}/train.csv")
test = pd.read_csv(f"{d}/test.csv")
if PARITY_FOLD >= 0:
    from sklearn.model_selection import GroupKFold
    fold = np.empty(len(train), dtype=int)
    for k, (_, va) in enumerate(GroupKFold(5).split(train, train.model, train.prompt_id)):
        fold[va] = k
    test = train[fold == PARITY_FOLD].reset_index(drop=True)
    train = train[fold != PARITY_FOLD].reset_index(drop=True)
train_text = train.text.fillna("").astype(str).tolist()
test_text = test.text.fillna("").astype(str).tolist()
models = train.model.to_numpy()
classes = sorted(train.model.unique())
assert len(classes) == 18, f"expected 18 models, got {len(classes)}"
stage("data", train=len(train), test=len(test), parity_fold=PARITY_FOLD, design=CONFIG["design"], mode=CONFIG["mode"])

Xtr = build_X(featurize(train_text), find_groups(train_text), CONFIG["mode"])
g_test = find_groups(test_text)
Xte = build_X(featurize(test_text), g_test, CONFIG["mode"])
stage("features", columns=Xtr.shape[1], test_groups=len(np.unique(g_test)),
      test_multi_groups=int((pd.Series(g_test).value_counts() >= 2).sum()))

if CONFIG["design"] == "flat":
    cfg = CONFIG["flat"]
    m = make_model(cfg["kind"], cfg["params"], n_jobs=N_JOBS)
    m.fit(Xtr, np.array([classes.index(v) for v in models]))
    assert list(m.classes_) == list(range(18))
    P18 = np.clip(m.predict_proba(Xte), 1e-6, 1)
    P18 /= P18.sum(1, keepdims=True)
else:
    fitted = {n: fit_node(n, c, Xtr, models, classes, n_jobs=N_JOBS) for n, c in CONFIG["nodes"].items()}
    P_top = predict_node(fitted["top"], Xte)
    P_sub = {n: predict_node(f, Xte) for n, f in fitted.items() if n != "top"}
    P18 = combine(P_top, P_sub, classes, CONFIG["use_cg"])
assert P18.shape == (len(test), 18) and np.isfinite(P18).all()
stage("trees", rows=len(test))

z = post(np.log(np.clip(P18, 1e-12, None)), g_test)
pred = np.array(classes)[z.argmax(1)]
sub = pd.DataFrame({"id": test.id.astype(str).values, "model": pred})
assert len(sub) == len(test) and sub.id.is_unique and sub.model.isin(classes).all()
sub.to_parquet(f"{OUT_DIR}/submission.parquet", index=False)
vc = sub.model.value_counts()
SUMMARY["prediction_counts"] = vc.to_dict()
if PARITY_FOLD >= 0:
    from sklearn.metrics import balanced_accuracy_score
    SUMMARY["parity_score"] = round(float(balanced_accuracy_score(test.model, pred)), 4)
    print("PARITY balanced accuracy", SUMMARY["parity_score"], flush=True)
stage("written", rows=len(sub), classes_predicted=len(vc), top_share=round(float(vc.iloc[0] / len(sub)), 3),
      post_applied=len(test) >= MIN_POST_ROWS)
print(vc.to_string(), "\nwritten", len(sub), "rows", flush=True)
