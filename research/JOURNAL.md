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

### e005 e002 config on run2final init (2026-11-09)
What: lr 1e-4 / head 1e-3, 2 epochs, init `run2final` (stage 1 finished, 1,875 steps), folds 0 and 1. Why: it is the init
that ships (queue idea 7); re-run of an earlier config is justified by the changed init.
Result: f0 raw 0.567 / sinkhorn 0.570, f1 0.548 / 0.574. Mean raw 0.557 (same config on run2s1000: 0.539, +1.8),
sinkhorn 0.572. 25 min.
Decision: new baseline for all further tests: init run2final, lr 1e-4 / 1e-3, 2 epochs, raw 0.557 on folds 0+1.
Next: e006 adds 3,000 random hostlike rows to the stage-2 training set (queue idea 5; the full 6,897 rows would make
a session ~90 min, 3,000 gives ~50 min and tests the direction).

### Main agent note (2026-11-09 06:25)
Local stage-2 (e002 config, init run2final): fold 2 raw 0.5685, fold 3 0.5815; with e005 folds 0-1 the 4-fold OOF mean is 0.566. Blend s2 0.6 + s1 0.2 + svm 0.1 + lgbm 0.1: OOF raw 0.575, sinkhorn 0.580 (research/fit_blend_s2.py). Full fit s2all uploaded as s2all in ntl-qwen-weights. Main agent is testing seed averaging locally (seed 7: s2f2b, s2allb); research agent need not duplicate seed experiments.

### e006 add 3,000 hostlike rows to stage 2 (2026-11-09)
What: e005 config plus a random 3,000-row sample of hostlike.parquet (seed 42) appended to each fold's ~2.1k real rows
(~5.1k rows, 2 epochs, 320 steps). Why: queue idea 5; more host-like text might add data volume.
Result: f0 raw 0.535 / sinkhorn 0.544, f1 0.520 / 0.533. Mean raw 0.528 vs 0.557 for e005 (-2.9), 49 min vs 25.
Decision: discard. Hostlike text, as used here, drags stage 2 away from the real train.csv distribution (more rows of
a slightly different distribution cost accuracy rather than add it). I did not test a smaller share or a pre-training
mix; with this much of a loss I do not expect a small share to give the +1.0 needed.
Offline ensembles (CPU, log-softmax averages on the same folds): e005 alone 0.557; e005+e002 (other init) 0.558;
e005+e006 0.560; e005+e002+e004 0.549. No ensemble of stage-2 runs from this recipe gains >= 1 point, so checkpoint /
config averaging is not worth GPU time here. The plateau around 0.55 seems robust to lr, epochs, init mix and ensembling.
Next: e007 label smoothing 0.1, e008 batch 16 (more optimizer steps), then a seed repeat of the best to measure noise.

### Main agent note (2026-11-09 06:55)
LB: stage-2 blend (s2all 0.6 + s1 0.2 + svm 0.1 + lgbm 0.1, sinkhorn) = **0.550** (OOF 0.580). Stage 2 transfers to the hidden test; CV-to-LB gap about -0.03.

### e007 label smoothing 0.1 (2026-11-09)
What: e005 config with --label-smoothing 0.1. Why: queue idea 6; guards against overconfident logits on easy classes.
Result: f0 raw 0.563 / sinkhorn 0.556, f1 0.533 / 0.570. Mean raw 0.548 vs 0.557 (-0.9), sinkhorn 0.563 vs 0.572.
Decision: discard (not better, within noise but on the wrong side). 22 min.
Next: e008 effective batch 16 (accum 2, 268 steps instead of 134): the train loss at the end of stage 2 is still ~1.1,
so more, smaller updates at the same lr may fit better without more epochs.

### Main agent note (2026-11-09 07:05)
Seed averaging (fold 2, stage 2 seeds 42 and 7): raw 0.5685 / 0.5796 -> avg 0.5833; sinkhorn flat. s2allb (seed 7 full fit) uploaded. Local run3 = stage 1 continued from run2final on 29,729 unseen pool rows (lr 5e-5), then stage 2 on folds 2,3 (s3f2, s3f3) to compare with s2f2/s2f3.

### e008 batch 16 (accum 2) (2026-11-09)
What: e005 config with --accum 2 (effective batch 16, 268 steps). Result: f0 raw 0.554 / sinkhorn 0.565, f1 0.546 / 0.561.
Mean raw 0.550 vs 0.557 (-0.7), sinkhorn 0.563. 26 min. Decision: discard (flat).
Reading so far (e002..e008, init run2final or s1000): lr 5e-5..1e-4, 2-3 epochs, batch 16/32, label smoothing, ensembles
and extra hostlike all land at 0.53-0.56 raw on folds 0+1; only the init (run2final vs s1000, +1.8) and a bad lr (2e-4)
or off-distribution data (hostlike, -2.9) moved the number. Optimisation tweaks are exhausted, so the next ideas
target regularisation / data: e009 random-span crop augmentation (--crop-min 0.6), e010 higher LoRA/head dropout.
Both need new train_exp.py options (--crop-min, --lora-dropout, --head-dropout; defaults reproduce the old behaviour).

### e009 random-span crop augmentation (2026-11-09)
What: e005 config; each training passage is cut to a random contiguous span of 60-100% of its tokens (rows under 48
tokens untouched), new crop each time a row is seen. Why: stage 2 is data-limited (2.1k rows), crops give the model
different views of the same passage.
Result: f0 raw 0.557 / sinkhorn 0.552, f1 0.528 / 0.565. Mean raw 0.543 vs 0.557 (-1.5). 23 min.
Decision: discard. Shorter views remove style evidence and the eval passages are always full length.
Next: e010 = e005 with --seed 7. Not an improvement test: it measures run-to-run noise (same config, different seed:
batch order, LoRA dropout masks, head dropout) so every difference above can be read against it; and the logits give a
real two-seed ensemble test.

### e010 seed repeat of e005 (2026-11-09)
What: e005 config exactly, --seed 7 (different batch order and dropout masks). Purpose: noise floor.
Result: f0 raw 0.552, f1 0.544, mean 0.548 vs 0.557 for seed 42: a 0.9 point gap from the seed alone on a
two-fold mean. Predictions of the two seeds agree on only 79-80% of rows. Averaging the two seeds' log-softmax:
f0 0.557, f1 0.559, mean 0.558 (+0.5 over the mean of the single runs 0.553).
Consequence for everything above: the noise of a two-fold mean is about +-1 point, so e007 (0.548), e008 (0.550),
e009 (0.543), e004 (0.531 on s1000 init) and e002 (+0.8 over e001) are all within noise of the baseline; only
e003 (lr 2e-4, -2.3 vs e002 on s1000) and e006 (hostlike, -2.9) are clearly worse, and e005 vs e002 (init, +1.8, and
both folds moved the same way) is the only clear gain. Baseline raw is best stated as 0.553 (mean of two seeds).
The keep rule (>= 1.0) is about the noise size, so any single positive result needs a seed repeat before being believed.
Next: e011 dropout (LoRA 0.15, head 0.3), the last cheap regularisation test; if flat, the stage-2 hyper-parameters
are done and I move to something structural (pseudo-labelling the unlabeled evaluation rows, label-free so it is
legal in CV).

### e011 higher dropout (2026-11-09)
What: e005 config with LoRA dropout 0.15 (was 0.05) and head dropout 0.3 (was 0.1). Result: f0 raw 0.557 / sinkhorn
0.583, f1 0.537 / 0.570. Mean raw 0.547 vs 0.557 (seed-7 repeat of e005 gave 0.548), sinkhorn 0.577 (highest so far,
but within the same noise). Decision: discard, flat. This closes the stage-2 hyper-parameter sweep: nothing beats the
e005 recipe beyond noise.

### e012 plan: pseudo-labelling the unlabeled evaluation rows (2026-11-09)
Idea: the hidden test (~4,000 unlabeled rows) is available at inference time, so a self-training stage is legal. CV
simulation, label-free: take the e005+e010 averaged predictions on a fold's validation rows, keep the most confident 40%
(offline check on folds 0/1: top 30% is 86% correct, top 50% is 75% correct) with the argmax as label, add them to the
fold's training rows, retrain from run2final with the e005 recipe, score on the fold's full validation set. Limits: the
pseudo rows are the easy ones (little new information), and a 540-row fold gives only ~216 pseudo rows versus ~1,600 on
the real test, so CV likely understates any gain. Needs train_exp.py --pseudo/--pseudo-frac (dataset versioned) and the
runner's CODE:<file> argument resolver. Fold labels are never read to build the pseudo set.

### Main agent note (2026-11-09 08:55)
Kaggle stage-1 data runs (full val 2,160 rows): run2 (pool) 0.521, k_host1 (pool+hostlike) 0.528, k_host3 (pool+hostlike x3) 0.530. Stage-1 ensembles: run2+kh3 raw 0.541 / sinkhorn 0.554. Uploaded as kh1final, kh3final. Main agent runs stage 2 from kh3final (folds 0-3 + full fit) in kernel ntl-s2-main (main agent slot); research agent should not duplicate this.

### e012 pseudo-labelling (2026-11-09)
What: e005 recipe plus 216 pseudo-labelled rows (top 40% by confidence of the e005+e010 average on the fold's own
validation rows, argmax labels, no true labels read), 2 epochs. Result: f0 raw 0.552 / sinkhorn 0.550, f1 0.537 / 0.557.
Mean raw 0.544 vs 0.553 (mean of the two e005-recipe seeds). 27 min. Decision: discard; no gain at this scale. It is
untested at the real scale (~1,600 pseudo rows on the 4,000-row test) and the selected rows are the ones the model
already gets right, so I do not expect it to help, but CV cannot show that.

## Handoff (gen 1)
Budget used: 5.67 of 14 GPU hours (13 sessions). Nothing is running. Last push: e012 (finished and logged).
**Best config (baseline for shipping):** stage 2 from `run2final`, lr 1e-4 body / 1e-3 head, 2 epochs, bs 8 x accum 4
(eff. 32), warmup 0.03, maxlen 320, eval maxlen 384, fp16, otherwise train.py defaults. Folds 0+1 raw balanced accuracy
0.557 (seed 42, e005) and 0.548 (seed 7, e010): mean 0.553; sinkhorn 0.572 / 0.561. Stage 1 alone on the same folds
(run2s1000, zero updates) scores raw 0.459, so stage 2 adds about +7 points, and `run2final` as init adds about +1.8 over
`run2s1000`.
**Table (raw, folds 0+1 mean):** e001 s1000 baseline 5e-5 0.531 | e002 lr 1e-4 0.539 | e003 lr 2e-4 0.516 | e004 3 epochs
0.531 | e005 run2final lr 1e-4 0.557 | e006 +3k hostlike 0.528 | e007 label smoothing 0.548 | e008 batch 16 0.550 |
e009 crop aug 0.543 | e010 seed repeat 0.548 | e011 dropout 0.547 | e012 pseudo-labels 0.544.
**Findings:** (1) the seed noise of a two-fold mean is about +-1 point, so everything between 0.543 and 0.557 on run2final
is indistinguishable; the stage-2 hyper-parameter space around the public recipe is flat. (2) Clear negatives: lr 2e-4,
hostlike rows added to stage 2 (-2.9, so off-distribution text hurts stage 2). (3) Ensembling stage-2 runs adds little:
two seeds +0.5, mixed configs 0 to +0.3 (logit averages computed offline from research/runs/*). (4) After stage 2,
sinkhorn adds only ~+1 on average (vs +4 on stage-1 weights): stage 2 already fixes most of the class-prior skew;
residual errors are sibling models (gemini-3.7-flash vs 3.6/3.8, glm-5.3 vs flash, gpt-5.6 sol/luna/terra).
claude-fable-5 has recall 0.13 and gemini-3.7-flash 0.23 (e002 pooled folds).
**Recommendation to the main agent:** ship the e005 recipe from run2final; fit it on all of train.csv (cv-fold -2) and, if
time allows, average 2-3 seeds (about +0.5, cheap) rather than tuning further. The big levers are upstream of stage 2:
a better stage 1 (more or more host-faithful pool data; run2 val was flat while train loss fell) and more real host
rows. I did not test `train_regen_norm.parquet` (leakage rule); it is the one untested data source that could add real
host-style rows, but it can only be evaluated by a protocol that excludes the validation prompts, which I did not build.
**What I would try next if continuing:** (a) stage 2 starting from the k_host1/k_host3 stage-1 weights when they exist,
same recipe, to see if host-faithful stage 1 beats run2final; (b) stage 2 with train_regen_norm rows after removing the
rows of each fold's validation prompts (requires prompt ids in that file); (c) folds 2+3 only matter if a candidate
appears; none did, so I did not spend GPU on them.
**Open doubts:** folds 0+1 are 1,080 rows total; every number has +-1 noise. Fold 4 (lockbox) was never touched. All
`grouped*` columns in results.tsv were ignored for decisions. train_exp.py gained options --extra/--extra-n, --crop-min,
--lora-dropout, --head-dropout, --pseudo/--pseudo-frac (all default off); finetune/train.py is untouched. run_job.py
now accepts CODE:<file> args. Per-job val_logits.npy/val_ids.csv (and epoch snapshots for e003/e004) are in
research/runs/<exp>/<job>/.

### e013 leak-safe train_regen extras (main agent, 2026-11-09)
What: e005 config plus train_regen_norm rows (all 18 models re-answering the host prompts), with the scored fold's prompts removed. Result: raw 0.5185 / 0.5444 (mean 0.531, e005 0.557), sinkhorn 0.535 / 0.565, 46 min. Decision: discard. External rows hurt stage 2 even when topic-matched (same as e006 hostlike, -2.9). Keep train_regen out of the final fit too.

### Main agent note (2026-11-09 10:50)
LB: three stage-2 fits (run2, run3, k_host3 starts) 0.8 + SVM 0.1 + LGBM 0.1, sinkhorn = **0.555** (OOF 0.591; previous 0.550 at OOF 0.580). Same-backbone ensembling gives diminishing returns; bigger gains need a stronger or different backbone or more real host-like data (decision for the user).

### Adversarial ideas (main agent, 2026-11-09 13:00)
Duel specialists (per sibling group, logistic on PCA of frozen run3 features, pool + host rows, out-of-fold): lose to the blend inside every group at 32/128/512 dims (e.g. gemini flash 0.553 vs 0.592, claude 0.713 vs 0.748); overriding costs 0.3-2.4 pts. Adversarial booster (logistic on blend OOF log-probs + PCA features, errors up-weighted, cross-fitted): best 0.5898 vs 0.5889 (noise), more error weight is worse. Conclusion: the sibling ceiling is signal in the data/features, not the decision layer. Both closed.

### e014 hard-negative margin (main agent, 2026-11-09)
What: e005 + hinge 0.5 x max(0, 1 - (true logit - strongest rival logit)), i.e. "it is NOT that sibling" inside stage-2 training. Result: raw 0.5759 / 0.5444 (mean 0.560 vs e005 0.557), sinkhorn 0.585 / 0.567. Decision: discard (within seed noise).

### e015 FGM adversarial training (main agent, 2026-11-09)
What: e005 + FGM, input embeddings perturbed by eps 1.0 (global norm per batch) along the loss gradient, second loss on the perturbed pass. Result: raw 0.5259 / 0.5315 (mean 0.529 vs 0.557), sinkhorn 0.533 / 0.544, 43 min per fold. Decision: discard. Adversarial direction closed: duel specialists, booster, hard-negative margin and FGM all flat or worse; the ceiling is feature information, not decision pressure.

### Style-feature sibling re-ranker, v40-style (main agent, 2026-11-09)
Per-group logistic on our 316 style features (pool 26k sibling rows + host outside fold), mass-preserving re-rank, out-of-fold: within-group style alone 0.48-0.55 vs blend 0.59-0.75; blend + style at lam 0.1/0.25/0.5/1.0: 0.5875/0.5843/0.5787/0.5685 vs 0.5889. Discard: the style evidence is already in the blend through the LightGBM member.

### Notebook components, one by one (main agent, 2026-11-09 17:30)
1. Partial Sinkhorn: alpha 0/0.25/0.45/0.65/0.85/1 on balanced OOF rows 0.589/0.591/0.594/0.595/0.592/0.591; on unbalanced resamples 0.583/0.582/0.581/0.580/0.574/0.571. KEEP alpha 0.45 (near-best when balanced, cheap hedge when not).
2. OOF sibling-style residual, rebuilt as in the Who-am-I v40A code (family prefix experts on stylometry, reliability from inner OOF, disputed-row gate, max_alpha 0.08): 0.5903 argmax / 0.5917 with partial sinkhorn vs blend 0.5889 / 0.5940. Style experts reach only 0.39-0.42 within families. DROP.
3. Four-view n-gram SVM (word, char, shape, topic-masked): host-only alone 0.361, host + pool (2k/model, w 0.3) alone 0.452 vs old SVM 0.371. In the blend: best of a 12-point weight grid 0.5972 vs 0.5940 partial-sinkhorn (+0.3, selection noise); at the old weight 0.5880. DROP: Qwen already carries this information.

### LB: prompt centering (main agent, 2026-11-09 18:00)
Notebook v5 = three stage-2 fits + SVM + LGBM, prompt centering w 0.5, full sinkhorn: **LB 0.572** (previous 0.555). Between the OOF simulations (partial groups +0.9, full groups +4.9): the hidden test has real multi-model prompt groups, just not complete 18-model sets. Centering is now a core component; the shared-prompt Gaussian becomes more promising. Partial sinkhorn 0.45 added to the notebook for the next submission.

### Notebook component 4: shared-prompt Gaussian (main agent, 2026-11-09 18:40)
Two-way model x = class mean + prompt shift + noise, S and T diagonalised together, shift estimated per found cluster with shrinkage lam/(1+n lam), 5 EM rounds (features/prompt_gauss.py). Out of fold, with partial sinkhorn 0.45; baseline = blend - 0.5 x cluster mean:
- full groups: baseline 0.6306; + 0.2 x Gaussian on blend log-probs 0.6338; + 0.2 x Gaussian on frozen run3 PCA-128 0.6366 (alone 0.597 / 0.515).
- partial groups (2-10 per prompt): baseline 0.6016; + Gaussian 0.6012 / 0.6000.
DROP: +0.3 to +0.6 only when groups are complete, zero on partial groups; plain centering already captures the shared prompt shift.

### run4 stage 2 and notebook v6 (main agent, 2026-11-09 20:20)
run4 (stage 1 continued on 92k pool rows) final 0.568 vs run2/run3/k_host3 0.521/0.527/0.530. Stage 2 from run4 (s4ff0-3): 0.617/0.552/0.626/0.594, family raw 0.597 vs s2/s3/h3 0.566/0.573/0.572. With centering w0.5 + partial sinkhorn 0.45 over folds 0-3: v5 blend 0.6306; s4 weight 0.2/0.4/0.5/0.6/0.8 (old families share 0.8 - w) 0.6375/0.6417/0.6394/0.6403/0.6412: a flat plateau from 0.4 to 0.8, about +1 pt. Fold 0 alone had suggested +2.4 (0.639 -> 0.663); the 4-fold number is the one to trust. v6 submitted at 0.5 / 0.1 x 3 / SVM 0.1 / LGBM 0.1, partial sinkhorn 0.45.

### Forum read, critically (main agent, 2026-11-09 21:30)
Read 16 threads via Kaggle's internal API (the page itself is JavaScript-only). Facts (host or competition API):
- Public LB = 40% of the hidden test (~1,600 rows), private = 60%. The mix of models differs between public and private (Overview), so class balance is not uniform: keep Sinkhorn partial, not full.
- Host: using relationships between test passages (clustering, pseudo-labels) is allowed. Self-generated data within reasonable cost (well under the $5k prize) is allowed and need not be public; the Benchmarks proxy is not meant for training data. Public Kaggle datasets and public LoRA weights are fine.
- Test prompts do not all have 18 answers (Overview), consistent with our 0.367 group-assignment probe.
Participant claims worth testing (not accepted at face value):
- Open Pool v14 (223,840 rows) drops 284 mislabelled rows and adds gen_reasoning: only kimi-k3 rows with reasoning "high" look like the real kimi-k3 (25% vs 8-11% recognised by a train-only TF-IDF yardstick; within-source control supports it). Check our kimi-k3 recall and pool version before the next stage 1.
- Public notebooks score 0.65-0.66 with Qwen3-14B / Qwen3-8B / Llama-8B two-stage LoRA24 + a prompt-centred "twin" head trained on batches of same-prompt triplets (removal costs 0.006-0.017 LB), a Gaussian head, ATLAS (worth -0.009 on one base, +0.001 on another), n-gram SVC, Sinkhorn 0.30-0.45 better than 0.60 (+0.003). Training-time prompt centring is the idea we have not tried; our post-hoc centring gave +1.7.
- Stage 2 there uses lr 5e-5 / head 5e-4 (ours 1e-4 / 1e-3) and class-weighted stage 1.
- Member diversity: a member retrained on the same pool with higher OOF scored lower on LB (0.651 -> 0.647); a different data mix adds more than OOF suggests.
- fp16 on T4 with a growing LoRA can overflow, skipped batches cost 0.10 OOF in one report: check our Kaggle 8B run's skipped-batch count. Kaggle TPU v5e-8 reported 14x faster than 2xT4 for Qwen3-8B LoRA in bf16, but queue waits up to ~23 h.
- Base-LM likelihood features (GLM-4-9B, Qwen3-8B) reported -0.002 in one grouped-CV stack: low prior for ATLAS.
- Noise of a difference of two public scores: sd ~ sqrt(f / n_public), f = share of rows where exactly one is right; for f 10-20% and 1,600 rows, 0.008-0.011.

### Notebook component 5: ATLAS-style base-LM likelihood view (main agent, 2026-11-09 22:15)
Qwen3-4B base (4-bit, bf16 compute) per-token log-prob / entropy / top-1 stats -> 37 features (features/atlas.py), LightGBM on host (+2,950 pool rows) out of fold (features/atlas_eval.py). Alone 0.118 (chance 0.056): weak, but independent. Added to the v6 blend before centering + partial sinkhorn (baseline 0.6394): ungated w 0.1 0.6486, w 0.2 0.6519; gates add nothing. Per fold (w 0.1 / 0.2): 0.650 -> 0.652 / 0.661, 0.593 -> 0.607 / 0.609, 0.656 -> 0.667 / 0.672, 0.659 -> 0.669 / 0.665: up on every fold. KEEP, ungated, w 0.15 (middle of the plateau, not the grid best). To ship: compute the features in the notebook with the same dtype as on Kaggle (fp16 on T4; recompute local features in fp16 to match), LightGBM trained on all host + pool rows.
Shared-prompt Gaussian re-run after the error-check fixes, against the v6 blend: partial groups 0.6180 -> 0.6185 at best. Still DROP.
ATLAS follow-up (22:35): fp16 features (Kaggle T4 dtype) have 0 NaN, correlate 0.999 with bf16. Seed-averaged (5 LightGBM seeds, class-balanced): host+pool at w 0.15 gains +1.08 (fp16, range +0.97 to +1.20) / +1.15 (bf16); host-only only +0.1 / +0.4. The single-seed fp16 eval showed +0.3 because it picked host-only by its stand-alone score. Shipping: fp16, host+pool, balanced, 5-seed bag, w 0.15, ungated.

### Night of 2026-11-09 (main agent, 00:30)
- Kaggle 8B stage 1 finished (k8a pool 60k 0.540 vs 4B run2 0.521; k8b pool+hostlike 0.555 vs 4B k_host1 0.528) but the notebook hung in output saving (likely symlinked 8B+4B models copied into the output, over 20 GB); cancelled by the user, weights lost. Runner fixed (links removed before the end).
- v6 submission: ERROR after 4.5 h, "A system error. Please try resubmitting" (Kaggle infrastructure; leaderboard-wide scoring throughput dropped ~10x that afternoon).
- v7 (v6 + ATLAS, logs, staging, guards) passed two local parity runs and a final local run; Kaggle parity skipped to save GPU quota (user); version 9 queued, chain tools/submit_v7.sh submits after checking version 9's own output.
- Stage-2 lr 5e-5 A/B: fold 0 0.607 vs 0.617 at 1e-4: stopped, keep 1e-4.
- Local 8B: bs 8 spills past 8 GB (>30 s/step); bs 4 x accum 8 peaks 6.8 GB at 13.8 s/step. Overnight run k8b_local (pool_host1 67k, same recipe as Kaggle k8b) started 00:22, ~2,090 steps, ETA ~09:00.

### 8B stage 1 local and v7 queue (main agent, 2026-11-10 09:40)
k8b_local (Qwen3-8B, stage 1 on pool_host1 67k rows, 1 epoch, bs 4 x accum 8, 13.8 s/step, about 8 h): full val 0.5440
(class-balanced eval 0.5589). Same data on 4B (k_host1) gave 0.528, so 8B is +1.6 at equal data. run4 (4B, 92k more
pool rows on top of run2) is 0.568, so data volume still matters as much as model size.
Stage 2 from k8b_local started (tools/stage2_8b.sh: k8f0-3 then k8all, s4ff recipe, bs 4 x accum 8).
v7 (kernel version 9) sat QUEUED on Kaggle for 6 h (no GPU time reserved, quota 4 h left); the chain stopped without
submitting and was resumed (RESUME_V=9) instead of re-pushing, which would have re-queued it.
v7 submitted 2026-11-10 11:11 IST (kernel version 10, ref 58188309). Version 9 never left the queue (11.5 h, cancelled);
version 10 ran within minutes. Kaggle only accepts a version whose own run has finished and written submission.parquet
("Notebook is still running. Did not find provided Notebook Output File"). The CLI no longer prints "Successfully
submitted", so the chain now treats the submissions list as the proof.
v7 public LB 0.571 (v5 0.572). Out of fold, v7 vs v5 should be about +2 (s4 family +1, ATLAS +1.08); the LB shows 0.
Possible reasons: public-LB noise (a few hundred rows give an SE of roughly 1.5 to 2 pts, so +2 is inside it), or the
ATLAS member / s4 weight not transferring to the hidden test. One LB point cannot separate them; do not tune to it.
Mistake: v7 was submitted twice (refs 58188309 and 58188319, 26 s apart, both 0.571). A chain "stopped" from the
harness kept running in its bash child and submitted alongside the resumed chain. submit_v7.sh now holds a lock file.

### 8B stage 2 and the blend test (main agent, 2026-11-10 16:00)
k8f0-3 (8B stage 2 from k8b_local, s4ff recipe): 0.574 / 0.563 / 0.607 / 0.620, family raw 0.591 (s4 0.597).
research/blend_8b.py, folds 0-3, v7 post (centering 0.5 + partial sinkhorn 0.45), ATLAS rebuilt 5-seed as shipped:
v7 0.6491. Adding the 8B family on top: w 0.1 +0.42, w 0.2 +0.88 (4/4 folds up), w 0.3 +0.83 (4/4), w 0.5 +0.93 (3/4).
Moving weight from s4 to 8B is worse (0.35/0.35 +0.65, 0.25/0.25 +0.46). Pick: s4 0.5 + 8B 0.2. k8all (full fit) done.

### v8 on the leaderboard (2026-11-10, logged 2026-11-11)
v8 (v7 + 8B stage-2 member at 0.2, kernel version 11, submitted by the user) public LB 0.571, the same as v7 (0.571)
and v5 (0.572). The +0.88 out of fold did not show up on the public LB. User decision 2026-11-11: drop the 8B; the
local 8B base model and 8B weights were deleted (logs and logits kept).

### New direction (user, 2026-11-11): train.csv only
Two experiments, no external data:
1. Feature-tree pipeline (tree/ntl_tree.py, research/tree_cv.py): style features without word lists, prompt-centered
   features (z-score inside label-free prompt groups), company-first trees with a Claude+GLM tree, inverse-frequency
   weight on the company node, knob search over LightGBM / RandomForest / ExtraTrees.
   Smoke test: prompt-centering the features lifts a flat LightGBM from 0.293 to 0.380-0.387 (all 4 folds up).
   Parity: kernel and CV differed by 1 point from 1e-14 float dust in the features (LightGBM bins ties); fixed by
   rounding X to 9 decimals; LightGBM now averages 3 seeds. Kernel fold-0 parity then matches exactly (0.3667).
2. Qwen3-4B fine-tuned on train.csv only (tools/trainonly_4b.sh, runs t4f0-3 + t4all, s4ff recipe without --init).

### tree1 on the leaderboard (2026-11-11 22:00 IST)
Kernel ntl-tree v1 (company-first, centered style features, LightGBM/ET nodes, alpha 0.5): public LB 0.263.
CV folds 0-3 0.376, lockbox fold 4 0.345, so LB is 8-11 points under. The Qwen blend shows a similar CV-to-LB gap
(v7 OOF 0.649 vs LB 0.571), so part of it is the general shift between train.csv and the hidden test; the rest is
likely that centered-only features depend fully on the test's prompt groups. Company-level probe
(research/company_probe.py, logs/company_probe.log): every signal tops out near 0.60 company balanced accuracy
(style 0.574, centered chars 0.596, centered words 0.537, next-word stats + style 0.601).

### Qwen3-4B on train.csv only (t4f0-3, 2026-11-11 22:55)
s4ff recipe without stage 1 (no --init). Folds 0.402 / 0.389 / 0.424 / 0.376, OOF raw 0.398, after the notebook post
(prompt centering 0.5 + partial sinkhorn 0.45) 0.490, company balanced accuracy 0.673.
Same folds for s4ff (stage 1 on external pool, then stage 2): raw 0.597, post 0.640, company 0.749.
So external data is worth about +15 points after post; post helps the train-only model far more (+9 vs +4).
Train-only Qwen still beats every hand-made signal at company level (best probe 0.60).

### Shift-aware validation explains tree1 (2026-11-12 00:15)
Sources: Geirhos et al. 2020 "Shortcut Learning" (sent by the user); mizuno0522's public notebook ("test prompts
appear to have only a few answers each"; group assignment +0.06 CV, no LB gain; char+word TF-IDF LinearSVC LB 0.345);
Kaggle Data page: test has only id + text, unseen prompts, ~4000 rows, unequal and public/private-different model mix.
research/tree2_eval.py scores each fold twice: FULL (all 18 per prompt, as train.csv) and SPARSE (2-6 texts per prompt,
Dirichlet model mix, groups found on the subset, 3 draws). tree1's design: FULL 0.376, SPARSE 0.269, real LB 0.263.
So SPARSE is the first local number that predicts the LB; centering's +10 is a train.csv-structure shortcut.
SPARSE raw: style trees 0.26-0.27, + next-word averages 0.287, + signal 0.287, letter chunks (char_wb 2-5 TF-IDF + LR)
0.304, blend 0.3 tree + 0.7 letters 0.321 (pick is optimistic). Company under SPARSE about 0.55 at best.
Next-word "signal processing" features (features/signal_feats.py: GLTR ranks, FFT of surprise, autocorrelation,
runs): company 0.274 raw / 0.337 centered alone, +0 on top of style + next-word averages.
Checker issues on tree2_eval: SPARSE subsets (~120 rows) are below MIN_POST_ROWS so post never runs there; pooling
the 4 folds' sparse subsets into one ~480-row set is the fix.

### t4 on the leaderboard (2026-11-12)
Kernel ntl-t4 v1 (Qwen3-4B fine-tuned on train.csv only, single member + notebook post): public LB 0.434
(OOF post 0.490, so -5.6). Compare: tree1 0.263, old TF-IDF SVM 0.348, mizuno's char+word TF-IDF SVC 0.345,
v8 with external data 0.571. Train-only Qwen beats every train-only classical model by about +9 on the LB;
external data is still worth about +14 on the LB.

### t4 inside the best blend (2026-11-12)
research/blend_t4.py (copy of blend_8b.py, 8B family swapped for t4f0-3), folds 0-3, v7 post. v7 0.6491.
Replacing s4 with t4 (0.5): 0.6167 (-3.24, 0/4 folds up). Adding t4 on top of s4: +0.19 to +0.23 at w 0.1-0.3,
2-3/4 folds up, below the +0.5 keep bar (and the 8B's +0.88 out of fold gave 0 on the LB). Not submitted.
Estimated LB if swapped in: about 0.54 (v7 0.571 minus the OOF drop).

### Was t4 weak only because the blend was tuned to s4? Train-only blend (2026-11-12)
research/blend_trainonly.py, folds 0-3, notebook post, weights by coordinate ascent, nested (chosen on 3 folds).
- v7 as shipped 0.6491 (SPARSE post 0.601). v7 members re-tuned with s4: nested 0.6491.
- Q1, v7 members with t4 in place of s4, re-tuned: nested 0.6426; the tuner gives t4 0.05 and leans on s2mix. So it
  was not tuning: t4 is the weakest member (0.398 alone vs s4 0.597).
- Q2, every member train.csv only (t4, svm4 host, style LGBM cv, ATLAS host-only, letters LR): nested 0.5236,
  SPARSE post 0.483, vs t4 alone + post 0.4866 (+3.7). Letters and SVM carry the gain. Estimated LB about 0.47.
- TRACE fingerprints (arXiv 2603.28054: rank / entropy / surprise transition pairs, LR and the paper's 1-NN to one
  reference per LLM): alone 0.117 / 0.119, blend +0.05. Short texts (50-330 tokens) are far below the ~10K tokens
  the paper needs for a stable fingerprint. Dropped.
- Checker: no leakage or alignment bugs; issue 88 (SPARSE uses in-sample weights) and 89 (second-order stacking leak
  in nested weights, < 0.005) are caveats, not errors.
Conclusion: train.csv alone tops out about 12 pts below the external-data blend; the data, not tuning, is the gap.

### Calibrated hidden-test simulator (2026-11-12 03:00-04:15 IST, user asleep, autonomous)
research/test_sim.py: simulated tests drawn from the folds 0-3 OOF rows; model j answers a prompt with probability
min(1, m w_j), w ~ Dirichlet(alpha); groups found blind exactly as the notebook does. Fitted to three paired LB probes
(same members, post changed): P1 assignment -3.5, P2 full sinkhorn on TF-IDF -0.4, P3 centering on 3 Qwen families
+1.7. Sigma per probe from the simulated flip fraction on 1600 rows plus Monte Carlo variance; 30 draws per cell,
common random numbers. Checker issues 90-94 fixed or documented.
- Full train.csv structure reproduces our OOF (assignment +10, centering +5.3), so the sim is faithful to our CV.
- Posterior: about 4 passages per test prompt (bimodal: 1.5-2 or 4-6), mix fairly even (alpha >= 2: 85%).
- v7 under the posterior: raw 59.0, shipped post 59.5 (+0.5 only, vs +5.7 on full OOF). LB 0.571. So the LB is
  set by raw model quality; post-processing is worth about +1 at most on this test.
- Best robust post for v7: centering 0.25, clusters N/9, partial sinkhorn 0.45: +0.80 (SE 0.08), positive in every
  cell with posterior >= 0.05 (worst +0.45). Lighter settings (w 0.25, a 0-0.2, N/5) reach +1.3 but lose in some cells.
- research/test_sim_p4.py, v7 vs v5 (LB -0.1) as a diagnostic only (it changes members, checker issue 97): the sim
  predicts +3.8 under P1-P3 (about +2.4 from partial vs full sinkhorn, +1.4 from s4/ATLAS). The 3.9-point gap
  (about 4 sigma) means v7's changes did not reach the hidden test: the mix may be closer to even than P2 says,
  and/or the s4/ATLAS member gains shrink on the hidden test.
- v9 (tools/build_v9_kernel.py): v7 members + the robust post. Also a clean LB probe of post alone vs v7.
Local GPU: s4 seed ensemble started (tools/s4seeds.sh, seed 7, folds 0-3), with RAM guards (start >= 4 GB free,
pause if < 1.5 GB twice). The 29.5k unseen pool rows were not used for another stage 1: 8 of 18 models have none.

### v9 on the leaderboard (2026-11-12 04:30 IST)
v9 (v7 members; centering 0.25, clusters N/9, partial sinkhorn 0.45; kernel ntl-v9 version 1, ref 58251936):
public LB **0.580**, new best (v5 0.572, v7 0.571). The simulator predicted +0.80 vs v7 for this exact change; the LB
gave +0.9. First out-of-sample confirmation of the calibrated simulator. v9 - v7 is a clean post-only probe (P5).

### Simulator refit with P5 and a finer post grid (2026-11-12 05:00 IST)
research/test_sim_v2.py: P5 = v9 - v7 = +0.9 (clean); simulated +0.80 under the posterior, so the fit holds.
Posterior unchanged: ~4 passages per prompt, mix fairly even. 96 settings (centering 0.1-0.35, sinkhorn 0.2-0.6,
clusters 5-12) vs v9: best w0.35_a0.2_g5 +0.73 but -0.77 in the worst plausible cell (lighter sinkhorn wins when the
mix is uneven, loses when it is even). Decision rule (>= +0.75 and never worse; checker issue 100 selection-bias
margin): KEEP v9. Checker issue 99 (merge crash after simulation) fixed; analysis rerun from saved draws.

### s4 seed 7, exclusion post, per-model recall (2026-11-12 05:00-06:00 IST)
- s4 seed 7 (tools/s4seeds.sh, same recipe, init run4): folds 0.632 / 0.580 / 0.613 / 0.596 vs seed 42
  0.617 / 0.552 / 0.626 / 0.594; OOF raw 0.605 vs 0.597, mean of both 0.601. On the simulator with the v9 post
  (research/sim_members.py): +0.30 (SE 0.06), worst plausible cell -0.06. Below the +0.5 bar alone; seed 123 queued.
- Mean-field exclusion inside small clusters (research/sim_exclusion.py): up to +1.09 expected (clusters of 3, lam 1)
  but every strong setting loses 0.3-1.4 in some plausible cell; checker issues 101-102 add real-scale purity and
  duplicate-model risks. NOT adopted.
- v9-post recall per model on test-like draws: weakest glm-5.3 0.31, claude-fable-5 0.34, gemini-3.7-flash 0.38,
  gpt-5.6-sol 0.41, glm-5.3-flash 0.42, kimi-k3 0.43. The glm-5.3 / glm-5.3-flash / claude-fable-5 triangle confuses
  mostly within itself (the user's Claude/GLM idea), gemini-3.7 vs 3.8 flash, gpt-5.6 sol/terra/luna, kimi vs gpt-5.5.

### s4 seed 123 and the 3-seed average (2026-11-12 06:20-08:10 IST)
- s4 seed 123 (same recipe, init run4): folds 0.628 / 0.576 / 0.630 / 0.574 (seed 7: 0.632 / 0.580 / 0.613 / 0.596,
  seed 42: 0.617 / 0.552 / 0.626 / 0.594). Full-structure OOF raw of the members blend: s4x3 0.5977, same as s4x2.
- Simulator with the v9 post (research/sim_members.py, candidates fixed in advance): s4x2 +0.30 (SE 0.06, worst
  plausible cell -0.06); s4x3 +0.50 (SE 0.06, worst plausible cell -0.10).
- Decision rule (>= +0.5 and never worse in a plausible cell): s4x3 fails on the worst cell, and it sits exactly on the
  bar while the P4 diagnostic says member gains shrink on the hidden test. KEEP v9. Not spent: the s4alls7/s4alls123
  full fits for the notebook, a weights-dataset upload, about 2x the s4 inference GPU time on Kaggle, a submission.
- Seeds help with diminishing returns (+0.30 for 2, +0.50 for 3). If the bar is relaxed later, s4x3 is the ready next
  candidate: the s4alls7 full fit exists; s4alls123 still needs about 35 min on the local GPU.

### Kaggle vs external data notebook; v9 without the hostlike member (2026-11-12 IST)
- research/data_compare.ipynb (built by research/make_data_compare.py): Kaggle vs Pool 92k / Pool 60k / hostlike,
  class-matched. Pools close on average (1.5% of feature x model cells drift > 0.5 sd; hostlike 4.6%); tell-apart AUC
  split by prompt 0.76-0.77 pools, 0.83 hostlike (style). Style-only classifier, 120 texts/model: Kaggle-taught 33.3%
  vs 25.0 / 24.0 / 23.4. Hard sibling pairs have weak style clues even within Kaggle (split-half 0.10-0.32) and the
  external sets carry them about as well; clues lost outside: gemini-3.1-pro, gpt-6-astra. Checker issues 106-112 fixed.
- Lineage: s4 = run2 (pool 60k) -> run3 (+30k) -> run4 (+92k) -> train.csv: no hostlike. s2, s3 also none. Only h3
  (k_host3: pool + hostlike x3) saw hostlike (weight 0.1 in v9).
- research/sim_nohost.py: h3 removed, weight to s2+s3 (clean): -0.19 (SE 0.04), worst plausible cell -0.62; weight
  dropped: -0.11. h3 alone OOF 0.572 = s3 0.573 > s2 0.566. KEEP v9 with h3. Checker issues 113-115 fixed.
- research/sim_t4_for_h3.py: h3's seat given to t4 (train.csv-only Qwen, OOF 0.398 alone vs h3 0.572): -0.13
  (SE 0.05), worst plausible cell -0.46; t4 added on top with h3 kept: +0.06 (worst -0.34). KEEP v9. Checker issue 116 fixed.
