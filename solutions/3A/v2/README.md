# 3A · Vietnamese forest bird calls — v2 (push to the limit)

**Final v2 (score.py): public = 0.9738, private = 0.9721** · v1 = 0.9636 / 0.9627 · gold line 0.94 · approx. ceiling ≈ 0.979.

Files: `solution.py` (end-to-end, device auto-detect), `solution.ipynb` (Data / Model / Training & Inference / Predict walk-through),
`common.py` (mel caches, features, models, metric), `mix.py` (soundscape synthesis, extended from v1), `train.py` (one model + its two
validation AUCs + TTA test probabilities), `predict.py` (blend → CSVs / pseudo-labels), `public_submission.csv`, `private_submission.csv`.
`python solution.py` reproduces the pipeline (`CACHE_3A` = cache dir, default `./.cache`; `V1_3A` = folder with the v1 CSVs, default `..`).
`python solution.py --quick` is a 2-epoch smoke run (run end-to-end on aorus-ts, see bottom). No checkpoints are kept here.

Where it ran: every model was trained on **aorus-ts** (GTX 1050 Ti shared with four other agents, ≤ 2.4 GB VRAM for this job, 8 CPU threads);
ResNet18 ≈ 40 s/epoch there under contention (≈ 28 min per 40-epoch model), SED-CNN ≈ 3 s/epoch on the GPU or ≈ 50–60 s/epoch on 2–3 CPU threads.

## Headline
| step | public | private |
|---|---|---|
| v1 final ensemble (5 light SED-CNNs) | 0.9636 | 0.9627 |
| **ImageNet ResNet18 instead of the light CNN** (single, hold-out, round 1) | 0.9709 | 0.9684 |
| + all focal clips + pseudo-labelled real test backgrounds (single) | 0.9729 | 0.9724 |
| **final family blend** (3 ResNet18 + EfficientNet-B0 + SED-CNN) | 0.9738 | 0.9721 |

## 1. Data
* **What the test is.** 800 passive-recorder soundscapes (Cúc Phương / Cát Tiên, ~50/50) vs 1,300 loud focal clips (84 % Cúc Phương);
  birds in the test are quiet and sit in real noise: insects, frog choruses, rain, engine hum, non-target birds. 0–3 species per clip.
  The v1 data-generating model (mix denoised focal clips over test-derived floors + procedural distractors, or over real test clips with soft
  pseudo-labels) is kept — it is what made v1 work — and extended:
  * mixer is now mel-resolution-agnostic (64 or 128 mel; distractor widths & pitch shifts scale) and its RNG stream is identical for any
    resolution, so validation sets are the *same mixtures* for every front-end (blend studies across front-ends are valid);
  * optional random spectral-tilt EQ and partial-window gating of birds (tested, **not** used: hurt RV, see ablations).
* **Better validation (RV).** v1 validated only on synthetic soundscapes (SV). v2 adds **RV**: held-out focal birds (same 20 % stratified
  split as v1, never seen in training) inserted at −10…+12 dB into the **200 real test clips with the lowest v1 activity**
  (their real frogs/insects/rain/non-target birds are kept, labels = inserted birds only). 1,600 fixed mixtures each for SV and RV.
  RV is the harder and more realistic set (A0: SV 0.988 vs RV 0.974) and it ranked the backbones the same way the test did.
  Caveats found: (i) RV over-rates the light SED-CNN relative to the test (family blend looked +0.002 on RV, was ±0 on the test);
  (ii) models trained with pseudo-labelled test backgrounds are penalised on RV (they correctly fire on the real birds in the RV backgrounds,
  which RV counts as negatives) and full-data models have seen the RV birds → RV is only used to compare hold-out models.
* **Pseudo-labels / all inputs.** Round 2 uses *all* 1,300 focal clips plus the 800 unlabelled test clips as real backgrounds (50 % of mixtures)
  with soft labels `ps1 = 0.6·ResNet18(round 1) + 0.4·v1`, values ≤ 0.1 zeroed. Round 3 re-labels with `ps2 = 0.5·F1 + 0.3·A1 + 0.2·EffNet`.
* Things checked and dropped: clip-level logit normalisation (subtract α·clip mean logit to cancel “noisy clip ⇒ every species up”):
  RV 0.9736 → 0.9730 (α 0.1) → 0.9697 (α 0.5) ✗. Hour/site priors: hurt in v1 (−0.008 on the test), not revisited.

## 2. Model
* **ImageNet-pretrained torchvision backbones** (allowed by the rules) on a 3-channel log-mel “image” (128 mel, 50–4000 Hz, hop 16 ms):
  clip-median-centred log-mel, per-bin-median-removed log-mel, per-bin-80th-percentile-removed (clipped) log-mel.
  ResNet: stem max-pool removed (keeps 16-frame time resolution); output → frequency mean+max → the v1 **SED attention head**
  (frame logits; clip logit = ½(attention-pooled + max-frame)).
* Families: **ResNet18** (11 M), **EfficientNet-B0** (4 M), and the v1 **light SED-CNN** (1.2 M, 64 mel, 2 channels).
  ResNet34 was planned but did not fit the 2.2 GB per-process cap at batch 32 (cuDNN error) and there was no time left for a batch-16 run.
* Loss: BCE (focal loss was not better in v1). EMA of weights (0.999) is the model that is evaluated and saved.

## 3. Training & inference
AdamW (wd 1e-2), OneCycle (ResNet max LR 1e-3, EfficientNet/SED 2e-3), 40 epochs × 1,600 fresh mixtures (EffNet 30), batch 32, SpecAugment
(2 freq + 2 time masks), EMA, inference = mean of 4 circular time-shift views. Round 1 hold-out → pseudo-labels → round 2 on all data
→ round 3. Seeds give little (F1 vs F3 test-prediction correlation 0.996); families give diversity (ResNet vs SED-CNN 0.86).
**Final blend** (decided on RV before scoring it): family weights ResNet : EfficientNet : SED-CNN = 2 : 1 : 1 (best RV blend of the hold-out
models, prob-mean beat rank-mean on RV), ResNet family = F1_r18_ps, F3_r18_ps, F3_r18_ps2; EfficientNet = A2_eb0; SED = FS1_sed.

## Ablation log (own validation; 1,600 mixtures each, ± ≈ 0.002)
| run | change | SV AUC | RV AUC | test pub / priv |
|---|---|---|---|---|
| v1 R1 | v1 recipe, light SED-CNN (v1's own 800-mix SV) | 0.9894 | – | 0.9576 / 0.9573 |
| v1 final | 5-model v1 ensemble | – | – | 0.9636 / 0.9627 |
| A0_sed | v1 recipe re-run + EMA weights (hold-out) | 0.9884 | 0.9736 | – |
| A3_sedaug | A0 + spectral-tilt EQ (σ 0.3) + partial-window gating (p 0.3) | 0.9884 | 0.9713 ✗ | – |
| P1_sed | A0 + v1 pseudo-labelled test backgrounds (hold-out) | 0.9874 | 0.9710 (RV biased, see §1) | – |
| A0 + clip-norm | subtract 0.1 × clip-mean logit | 0.9885 | 0.9730 ✗ | – |
| **A1_r18** | **ImageNet ResNet18, 128 mel, 3 ch** (hold-out) | 0.9938* | **0.9763** | **0.9709 / 0.9684** |
| A2_eb0 | ImageNet EfficientNet-B0, 30 ep (hold-out) | 0.9932 | 0.9729 | – |
| blend A1+A2 (1:1) | prob mean | – | 0.9772 | – |
| blend A1+A0 (2:1) | prob mean | – | 0.9778 | – |
| blend A1+A2+A0 (2:1:1) | prob mean / rank mean | – | **0.9785** / 0.9778 | 0.9702 / 0.9687 |
| F2_r34_ps | ResNet34 | – | – | did not fit the VRAM cap |
| **F1_r18_ps** | ResNet18, all clips + ps1 test backgrounds (p 0.5) | 0.9952† | 0.9767† | **0.9729 / 0.9724** |
| F3_r18_ps | same, seed 3 | 0.9960† | 0.9758† | – |
| F3_r18_ps2 | round 3: ps2 pseudo-labels | 0.9962† | 0.9764† | – |
| FS1_sed | SED-CNN, all clips + ps1 | 0.9897† | 0.9749† | – |
| **final** | family blend 2:1:1 (above) | – | – | **0.9738 / 0.9721** |

\* A1's SV was measured on an earlier draw of the 128-mel SV set (before the RNG alignment); RV is identical for every run.
† trained on all focal clips (incl. the validation birds) → optimistic; only reported for sanity.

## score.py calls (black box, 8 allowed, 5 used)
| # | submission | public | private |
|---|---|---|---|
| 1 | A1_r18 single (hold-out ResNet18) | 0.9709 | 0.9684 |
| 2 | round-1 blend A1:A2:A0 = 2:1:1 | 0.9702 | 0.9687 |
| 3 | F1_r18_ps single (all clips + pseudo-labels) | 0.9729 | 0.9724 |
| 4 | **FINAL** family blend (ResNet ×3 at 2/3 each, EfficientNet-B0 ×1, SED-CNN ×1; chosen on RV before this call) | **0.9738** | **0.9721** |
| 5 | report only, *not* used to change the submission: ResNet-family-only mean (F1, F3, F3_ps2) | 0.9748 | 0.9731 |

**Honest read of calls 4–5.** The ResNet-only blend scores +0.001 better than the submitted family blend on both splits: the RV
validation over-rated the SED-CNN / EfficientNet families (same lesson as call 2). Fair-play says select on own validation, so the
submission stays the RV-chosen blend; the next iteration should fix RV (see below) rather than the blend weights.

## v1 → v2 → ceiling
v1 0.9627 → v2 0.9721 private (ceiling ≈ 0.979): ≈ 58 % of the remaining gap closed.
What mattered: (1) a pretrained ImageNet backbone instead of a from-scratch light CNN (+0.011 vs the v1 single model, +0.006 vs the whole v1 ensemble — the biggest jump);
(2) all focal clips + real test clips as pseudo-labelled backgrounds (+0.004); (3) the real-background validation RV, which caught that
EQ/gating and clip normalisation do not help and picked ResNet18 over EfficientNet-B0.

## What limits further gains
* **Compute**: one shared GTX 1050 Ti (≈ 28 min per ResNet18 under contention). Not done for lack of time/VRAM: ResNet34/50,
  DenseNet121 or ConvNeXt-T backbones (batch 16 / gradient accumulation), 80+ epoch schedules, more pseudo-label rounds with
  sharper targets, more backbone families (seeds of one family add almost nothing: r = 0.996).
* **Validation fidelity**: RV is closer to the test than SV but still not the test (it over-rated the SED-CNN family). A better RV would
  draw backgrounds from clips the current ensemble labels empty with high confidence and insert birds at the SNR distribution
  estimated from test detections.
* **Label/teacher noise**: pseudo-labels inherit the frog/silver-pheasant and big-arch/magpie-robin confusions; the residual
  gap to ≈ 0.979 is mostly the hardest species (greyel, grehor1, silphe have the lowest RV AUCs: 0.938 / 0.963 / 0.962 for A0).
* Not tried: waveform-domain mixing, full 8 kHz STFT resolution, time-stretch, test-time adaptation of BatchNorm statistics.

## Reproduction note
`solution.py --quick` (2 epochs × 64 mixtures per model) was run end-to-end on aorus-ts (GPU) in a clean cache: all 8 training runs, both pseudo-label rounds and the final `predict.py` blend completed and wrote both CSVs (shape 400 × 11).
Full runs were executed step by step through the same `train.py` / `predict.py` commands (see `solution.ROUND1/2/3`, `FINAL`);
CPU-trained SED runs used `--workers 0/1` (different mixture RNG streams than the default 3 workers).
Remote scratch (`aorus-ts:~/olympiad_offload/3A_v2/`) is cleaned after copying predictions back.
