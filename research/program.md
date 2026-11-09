# Autoresearch program: Name That LLM

You are the research agent. You run experiments on Kaggle's free 2x T4 GPUs, keep what helps, discard what
does not, and write down everything so that nobody (including a later you) repeats work blindly.
The main agent monitors you and owns submissions, the local GPU and all merges.

## Goal
Raise balanced accuracy on the hidden test (about 4,000 rows, 18 models). Public LB now: 0.402 (TF-IDF + LightGBM).
Rank 50 is about 0.692, the top is 0.752. Our main lever is the fine-tuned Qwen3-4B (multi-scale LoRA, 24 layers).
Local CV maps to LB roughly as CV - 0.02 so far.

## What already exists (read before acting)
* `finetune/train.py`: the trainer (read it once fully). Stage 1 = external pool, stage 2 = host train.csv with
  `--cv-fold k` (GroupKFold by prompt, 5 folds) and `--init <weights.pt>`.
* `research/code/train_exp.py`: YOUR copy of the trainer. Change this one, never `finetune/train.py` (a live
  local run resumes from that file). `research/code/assign.py`: group assignment (see below).
* `kaggle_research/run_job.py`: Kaggle runner. Set `EXP` and `JOBS` (one job per GPU), push, wait, fetch
  `summary.json`. It reports raw, sinkhorn, grouped and grouped_oracle balanced accuracy per job.
* Datasets (private, varadarajborkar/...): `ntl-research-code` (your train_exp.py + assign.py),
  `ntl-qwen-weights` (init weights as `<name>__weights.pt`: `run2s1000` now, `run2final` once run2 ends),
  `ntl-train-data` (pool_typo_60k.parquet, hostlike.parquet), plus the Qwen model and competition data.
* `research/JOURNAL.md` (narrative, newest last) and `research/results.tsv` (one row per experiment).

## Key facts
* The metric you optimise is **`raw`** (plain argmax balanced accuracy, mean of the screened folds). Report
  `sinkhorn` as context only: it assumes a balanced test set and has not been validated on the LB.
* Group assignment is a dead end: every train.csv prompt is answered once by each of the 18 models and that
  structure lifted OOF from 0.419 to 0.520, but the LB fell from 0.402 to 0.367 with it, so the hidden test is
  built differently. Ignore `grouped` / `grouped_oracle` in summary.json for decisions.
* Stage-2 recipe from public notebooks: lr 5e-5, head lr 5e-4, 2 epochs, reported about +9 points over stage 1.
* T4 has no bf16: always `--compute-dtype fp16` (tested: same accuracy, 4x faster). About 9 s per optimizer
  step at batch 32. A stage-2 fold (about 2,160 train rows, 2 epochs) is about 20-25 minutes.

## Protocol (the rules that keep results honest)
1. **Fold 4 is the lockbox.** Never use `--cv-fold 4`, never evaluate on fold 4, never read its scores.
2. **Screen on folds 0 and 1** (one per GPU, same config). Score = mean `raw` of the two folds.
3. **Keep rule:** a change is a candidate if it beats the current best by >= 1.0 point on folds 0+1 (one fold is
   about 540 rows: noise is about +-2 points per fold). Confirm a candidate on folds 2+3 before adopting it as
   the new baseline. A change that is flat (within 1 point) and adds complexity or time: discard.
4. **Never fake or massage numbers.** Report exactly what summary.json says. A crashed or partial run is a
   crash, logged as such. Never train on validation rows of the fold you score.
5. **No leakage:** `train_regen_norm.parquet` re-answers the host train prompts, so it may only be used in a
   final fit, never in a scored experiment. hostlike.parquet is clean (no host prompts).
6. **Before every experiment, read results.tsv** and check it is not a repeat. Re-running something is allowed
   only when the situation changed (new init weights, a bug fix, a different base config), and you must write
   why in the journal.
7. **One idea per experiment** where possible, so results can be attributed.

## Budget and etiquette
* GPU budget: at most 14 Kaggle GPU hours in total for you this week, unless the main agent raises it. Track the
  running total in results.tsv (column gpu_h) from `total_minutes` in summary.json.
* The account allows only 2 concurrent Kaggle GPU sessions. Before every push, check for `research/GPU_HOLD`:
  if it exists, the main agent needs the slot, so wait until it is gone.
* Run at most one Kaggle session at a time (kernel `ntl-research`). The kernel `ntl-qwen-train` belongs to the
  main agent; never push to, cancel or modify it, nor `ntl-qwen-blend`, `ntl-qwen-parity`, `ntl-tfidf-lgbm-blend`.
* Never submit to the competition. Never use the local GPU (a training run owns it). Local CPU work: at most
  6 threads, low priority.
* Waiting: start a background Bash loop that polls `kaggle kernels status varadarajborkar/ntl-research` every
  60 s and exits on COMPLETE/ERROR/CANCEL, then fetch outputs with `kaggle kernels output ... -p <scratch dir>`.
  `summary.json` is in the output folder. Do not poll in a tight loop; use the notification.
* When you change train_exp.py or assign.py: `cd research/code && kaggle datasets version -p . -m "<why>"`,
  then wait until `kaggle datasets status varadarajborkar/ntl-research-code` says ready before pushing a job.
* Commit to the `dev` branch after each experiment (journal, results.tsv, code). Commit messages must not
  mention Claude or AI and must not include Co-Authored-By lines. Never use em dashes anywhere. Push to origin dev.

## Ideas queue (start at the top; add your own, reorder as evidence comes in)
1. e000 smoke: the runner as committed (64 rows, 1 epoch) to prove the pipeline end to end. Not a result.
2. e001 baseline stage 2: `--init INIT:run2s1000 --lr 5e-5 --head-lr 5e-4 --epochs 2`. Also record what
   stage 1 alone scores on these folds (`--epochs` tiny is not the same; instead evaluate the init weights with
   `--n-train 8 --epochs 1 --lr 1e-9 --head-lr 1e-9` as a near-zero-update control).
3. Learning rate: 1e-4 / 1e-3; 2e-5 / 2e-4.
4. Epochs: 1, 3 (with the best lr).
5. Add hostlike.parquet to stage-2 training (needs a small train_exp.py option, e.g. `--extra <parquet>`).
6. Label smoothing 0.1. LoRA dropout. Max length 384 for training.
7. When `run2final` appears in ntl-qwen-weights (the main agent adds it after run2 finishes, about 04:15),
   re-run the best config with it: it is the init that will actually ship.
8. Checkpoint ensembles (evidence: run2 step 1000 + 1250 logit average on the 900-row subset: raw 0.480 vs
   0.463 / 0.430 alone, sinkhorn 0.520 vs 0.502 / 0.496). In stage 2, try averaging log-softmax of the end of
   epoch 1 and epoch 2, or of two seeds, scored on the same fold. Main agent is saving run2 weights at steps
   1000, 1300, 1500, 1700 and final (names run2s<step> in ntl-qwen-weights once uploaded).
9. Ensembling hint: keep `val_logits.npy` of good runs (the output folder has them) under
   `research/runs/<exp>/<job>/` (gitignored is fine via data/; or keep small npy files in research/runs) so the
   main agent can test blends.

## Hand-off
You will be retired near 170k tokens of context. Before that (or when the budget is spent, or the queue is
empty and you have no good idea left), write a `## Handoff (gen N)` section at the end of JOURNAL.md: current
best config and score, what is running, what you would try next and why, any open doubts. Then stop and
return a short summary to the main agent.
