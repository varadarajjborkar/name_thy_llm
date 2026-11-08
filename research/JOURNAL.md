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

### e002 lr 1e-4 / head 1e-3 (2026-11-09)
What: e001 with lr doubled (body 1e-4, head 1e-3), everything else equal. Why: queue idea 3; train loss was still
~1.1 at the end of e001 so a larger step may help.
Result: f0 raw 0.541 / sinkhorn 0.561, f1 0.537 / 0.570. Mean raw 0.539 vs baseline 0.531 (+0.8, below the 1.0 keep
rule), mean sinkhorn 0.566 vs 0.533 (+3.3, but sinkhorn is not the decision metric). Offline average of the e001 and
e002 log-softmax outputs (same init, same folds): f0 0.543, f1 0.531, i.e. no ensemble gain, as expected from two
runs differing only in lr. Train EMA loss at step 120 was 1.09 (e002) vs 1.12 (e001): 2 epochs on 2.1k rows still
underfit, so more optimisation (lr and/or epochs) looks like the direction.
Decision: not yet a candidate on raw. Skipping lr 2e-5 (the trend runs the other way, it would only confirm it).
Next: e003 lr 2e-4 / head 2e-3 to find where the lr curve peaks, with per-epoch logits saved (--eval-every 67) so
an epoch-1 vs epoch-2 ensemble can be tested offline.
Note: results.tsv gpu_h_total was briefly mis-summed by my helper; fixed (cumulative total after e002 = 0.97 h).

### e003 lr 2e-4 / head 2e-3 (2026-11-09)
What: e001 with lr 2e-4 (head 2e-3), 2 epochs, plus epoch-1 logits saved (--eval-every 67).
Result: f0 raw 0.511 / sinkhorn 0.543, f1 0.520 / 0.556. Mean raw 0.516 (vs 0.539 at 1e-4, 0.531 at 5e-5), so the lr
curve peaks near 1e-4 (5e-5: 0.531, 1e-4: 0.539, 2e-4: 0.516). Offline: epoch-1 snapshot alone 0.419 / 0.448 (mid
schedule at high lr, weak), epoch1+epoch2 average 0.522 / 0.526 vs epoch 2 alone 0.511 / 0.520 (+1.1 but still
below e002 alone). Average of the final e003 and e002 outputs (different lr, same init): f0 0.541, f1 0.552, mean 0.546
(+0.7 over e002 alone, within noise).
Decision: discard 2e-4. Keep lr 1e-4 / 1e-3 as the working lr for the next tests (best so far, e002; its margin
over e001 is under the 1.0 rule so I treat lr as a flat 5e-5..1e-4 plateau and choose 1e-4 only because every
following test needs to pick one). Next: e004 epochs 3 at lr 1e-4 (train loss was still ~1.1 at the end).

### e004 epochs 3 at lr 1e-4 (2026-11-09)
What: e002 config with 3 epochs (201 steps, cosine over all 3), snapshots every 67 steps. Why: queue idea 4, loss was
still falling at the end of 2 epochs.
Result: f0 raw 0.528 / sinkhorn 0.541, f1 0.533 / 0.554. Mean raw 0.531 (e002 2 epochs: 0.539), sinkhorn 0.547.
Snapshots inside the run: step 67 / 134 / 201 = f0 0.494 / 0.511 / 0.528, f1 0.507 / 0.526 / 0.533. Averaging the last
two snapshots: 0.522 / 0.532 (worse than final alone); e004 final + e002 final: 0.541 / 0.541 (mean 0.541, +0.2 over
e002 alone, no real gain). 33 min per session instead of 26.
Decision: discard 3 epochs; 2 epochs stays (more epochs do not help raw, accuracy plateaus at ~0.53-0.54 with this
init and 2.1k rows). The limiting factor looks like data quantity or distribution, not optimisation.
Next: wait for the GPU slot (main agent asked for research/GPU_SLOT_FREE before any push). Then e005 = e002 config
on init `run2final` (the init that ships), then e006 = e005 + hostlike (--extra).

Aside (CPU only, e002 outputs pooled over folds 0+1): per-model recall ranges from 0.13 (claude-fable-5) to 0.82
(claude-opus-5). Under-predicted classes: claude-fable-5 (2.2% of predictions vs 5.6% expected), gemini-3.7-flash
(3.0%), gpt-5.6-sol (4.2%); over-predicted: glm-5.3-flash, gpt-5.6-luna, gemini-3.8-flash (~7.5%). Largest confusions:
gemini-3.7-flash -> gemini-3.8-flash (27), claude-fable-5 -> glm-5.3-flash (21), glm-5.3 -> glm-5.3-flash (18),
gpt-5.6-sol -> luna/terra. Sibling versions of one vendor are the hard part, which is where the sinkhorn gain on
raw comes from (a prior fix, not new information).

### Main agent note (2026-11-09 05:20)
LB: Qwen run2 (final+s1700) 0.8 + SVM 0.1 + LGBM 0.1: argmax 0.520, sinkhorn 0.521 (OOF 0.543 / 0.546). CV maps to LB at about -0.02; sinkhorn neutral on LB. Main agent runs the e002 config (lr 1e-4/1e-3, 2 epochs, init run2final) locally on folds 2 and 3 (runs s2f2, s2f3) and a full fit s2all, to complete OOF with e005 folds 0-1.
