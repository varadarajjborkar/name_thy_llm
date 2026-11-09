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
