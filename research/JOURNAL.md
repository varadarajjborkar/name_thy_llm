# Research journal

Newest entries last. Each entry: date/time, experiment id, what and why, result, decision.

## Background (written by the main agent, 2026-11-09)
* TF-IDF word+char LinearSVC: grouped CV 0.368, LB 0.348.
* LightGBM on 316 style features, trained on 100k external passages: 0.365 on train.csv (unseen).
* Blend 0.5/0.5 (T=3): CV 0.419, LB 0.402. Sinkhorn class balancing: 0.422.
* Group assignment on that blend (research/assign_eval.py): oracle prompt groups 0.526, found groups 0.520.
  LB probe 2026-11-09: **0.367 vs 0.402 without it. Dead end**: the hidden test is not built like train.csv.
  Decide on `raw`. Sinkhorn on the same blend: LB 0.398 vs 0.402 (OOF gain was only +0.3 there, so this is
  inconclusive for Qwen, where sinkhorn adds about 5 points on validation). To be settled with a Qwen submission.
* Qwen3-4B base, QLoRA r=16, multi-scale head (mean pool at depths 2..24, 24 layers kept). Stage 1 `run2` on
  pool_typo_60k (60k external passages, typography-filtered), lr 1e-4 / head 1e-3, warmup 0.1, 1,875 steps,
  local RTX 4060. Validation = train.csv minus fold 4, 900-row stratified subset:
  step 500 raw 0.400 / sinkhorn 0.452; 750 0.393 / 0.454; 1000 0.463 / 0.502. Finishes about 04:15.
* Kaggle inference parity (run2 step 500 + LGBM, sinkhorn, same 900 rows): 0.464 on Kaggle T4 fp16, matches local.
* Kaggle ntl-qwen-train (main agent): k_host1 = stage 1 on pool + hostlike, k_host3 = pool + hostlike x3,
  started about 02:10, about 6.5 h. Compare with run2 to learn whether host-faithful data helps stage 1.
* Dead ends so far: run1 (36-layer seqcls head) learned slower than multi-scale; bf16 on T4 is 4x slower than fp16.

## Research agent A, gen 1 (log)
Columns f0..f3 in results.tsv hold **raw** balanced accuracy per fold (primary metric since the group-assignment
LB failure); mean_grouped is kept for completeness only and is not used for decisions. gpu_h = session
`total_minutes`/60 (one Kaggle session = one GPU-hour clock, both T4s busy).

### e000 smoke (2026-11-09)
What: runner as committed, init `run2` weights, 64 train rows, 1 epoch (2 optimizer steps), folds 0 and 1, one per GPU.
Why: prove the pipeline end to end (dataset mounting, bitsandbytes install, 2 parallel jobs, summary.json, assign).
Result: both jobs exit 0, 4.0 min each, 4.2 min total session (0.07 gpu_h). Model load + tokenization + 540-row eval
fits well within that. raw 0.419 / 0.356, sinkhorn 0.428 / 0.426 (n_val 540 each). Not a result: 64 rows from an
init that already had a stage-1 head; it only shows the code path works.
Decision: pipeline OK. Next e001: stage-2 baseline with run2s1000 init (lr 5e-5, head 5e-4, 2 epochs), then the
near-zero-update control (--n-train 8, which yields 0 optimizer steps, i.e. exact stage-1 score) as e001c.

### e001 stage-2 baseline and e001c zero-update control (2026-11-09)
What: stage 2 on folds 0 and 1 from `run2s1000` (run2 step 1000), lr 5e-5, head lr 5e-4, 2 epochs (134 optimizer
steps, ~2,144 rows), fp16, bs 8 x accum 4, maxlen 320, warmup 0.03 (train.py default). Control e001c: same init,
`--n-train 8 --epochs 1`, which gives 0 optimizer steps, i.e. the pure stage-1 score on the same 540-row folds.
Why: need the stage-2 reference and to know how much of the score is stage 2.
Result (raw / sinkhorn): stage 1 only f0 0.461 / 0.504, f1 0.457 / 0.496 (mean raw 0.459, sinkhorn 0.500).
Stage 2 baseline f0 0.528 / 0.522, f1 0.533 / 0.544 (mean raw 0.531, sinkhorn 0.533). So stage 2 adds +7.1 points
raw (public notebooks claimed ~+9), and after stage 2 sinkhorn no longer helps (+0.3 vs +4.1 on stage 1 weights):
stage 2 already fixes the class-prior mismatch. 25 min per session (0.42 gpu_h); control 2 min.
Decision: e001 is the baseline (raw 0.531). Added `--extra/--extra-n` to train_exp.py (dataset versioned) for the
hostlike idea. Next: e002 lr 1e-4 / head 1e-3.
