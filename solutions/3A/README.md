# 3A · Vietnamese forest bird calls — solution

**Final (score.py): public = 0.9636, private = 0.9627 → 🥇 Gold** (baseline ≈ 0.731 private; gold ≥ 0.94).

Files: `solution.py` (end-to-end), `solution.ipynb` (walk-through), `common.py` (paths, mel cache, features, model),
`mix.py` (soundscape mixture synthesis), `train.py`, `predict.py`, `prior.py` (hour prior, tested and *not* used),
`public_submission.csv`, `private_submission.csv`.

Run: `python solution.py` (GPU auto-detected; `CACHE_3A` sets the feature/checkpoint cache dir, default is a scratch dir outside this folder).
**Where it was trained:** first experiments on the local CPU (2 threads, very slow on a shared machine); all models in the final
ensemble and the ablations R1–R6 were trained on the remote host **aorus-ts** (GTX 1050 Ti, ≤1.1 GB VRAM per process, 3 CPU threads),
~6 min per model. Scoring was done locally with `score.py` (4 calls in total).

## 1. Data (≈60 % of the effort)
What the spectrograms showed (train vs public/private, plotted side by side):
* **Domain shift.** Train = loud, clean focal clips (one bird, high SNR). Test = passive-recorder soundscapes: birds ~4× quieter relative
  to the noise; stationary cicada/insect bands (2–3.8 kHz); pulsed insect tones (3–3.9 kHz); engine hum harmonics; rain; big harmonic
  arches (non-target birds); and a very common **frog/insect chorus** (regular pulse train, 1–3 harmonic dots) that the first model read as *silphe*
  (silphe "detected" in 47 % of night clips vs 7 % train prevalence) and arches read as *orimag1*.
* **0–3 species per test clip** vs mostly one in train (13 % of train clips have secondary labels).
* **Imbalance:** 230 orimag1 … 35 grehor1.
* **Metadata:** train hours imply strong diurnal priors (owl only at night, most species only 5–17 h). The model's own test predictions
  agreed for the owl (night 39 % vs day 2 %) but showed day species at night too. A soft hour prior **hurt** on the real test (0.957 → 0.949),
  so `hour`/`site` are *not* used.

What I did:
1. Cache linear mel power (64 mels, 50–4000 Hz, n_fft 512, hop 128) with each clip **RMS-normalised** (levels comparable).
2. **Denoise focal clips** (subtract per-bin median over time) → clean bird templates.
3. **Synthetic soundscapes in the mel-power domain**: background = stationary floor (20th pct per bin) of a random *unlabelled test clip*
   × random texture × slow gain drift; + Poisson(1.5) **procedural distractors** (insect tone, small/big harmonic arches, hum, rain, cicada band,
   clicks, frog chorus — the last two added after inspecting test false positives); + 0–3 bird templates (class-balanced √ sampling) with
   random time shift, ±2 mel-bin **pitch shift**, SNR −10…+12 dB; labels = union, **secondary labels** count as positives.
4. Features = 2 channels: clip-median-centred log-mel and per-bin-median-removed log-mel (cheap PCEN-like stationary-noise removal).
5. **SpecAugment** (2 freq + 2 time masks).
6. **Soundscape-like validation**: 20 % of focal clips (stratified) held out, 800 fixed mixtures of them over floors of *other* test clips than
   those used for training backgrounds. (A random focal split would be ~1.0 and meaningless.)
7. **Round 2 / pseudo-labels**: round-1 model labels the 800 unlabelled test clips (soft, p<0.1 → 0); round-2 models use *real* test clips as
   backgrounds 50 % of the time with those soft labels as targets, so they see the real frogs/insects/rain.

## 2. Model (≈30 %)
`BirdSED`: light VGG-style CNN (BatchNorm, 7 conv 3×3 layers, stride-2 stem, width 32→256, ~1.2 M params) → frequency mean+max pooling →
**SED attention head** (frame logits, softmax attention over time, averaged with max-frame logit). Designed small because the first runs
were CPU-only; width 16 is only marginally worse (0.9884 vs 0.9894 val).

## 3. Training & inference (≈10–20 %)
AdamW (wd 1e-2), OneCycle (max LR 2e-3), 40 epochs × 1600 freshly synthesised mixtures, batch 32, BCE (focal loss was not better);
checkpoint = best soundscape-val AUC in the second half; inference = 4 circular time-shift TTA; final = mean of 5 models:
R1 (hold-out), F1/F2 (all clips, 2 seeds), PF1/PF2 (all clips + pseudo-label backgrounds, 2 seeds).

## Ablation log
Soundscape-like validation AUC (800 held-out mixtures; from R1 on the val includes the frog/big-arch distractors). Test = score.py public / private.

| run | change | val AUC | test pub / priv |
|---|---|---|---|
| baseline notebook | plain CNN, random focal split | (≈0.99 on focal split, misleading) | ≈ – / 0.731 |
| B (CPU, w16, 2× time pool) | white-noise bg, no distractors, no denoise | 0.861 @ep6 (stopped) | – |
| C (CPU, same model) | full data recipe (test floors + distractors + denoise) | 0.943 @ep6, 0.969 @ep15 | – |
| A (CPU, w16, full res) | full recipe, 15 ep × 800 | 0.976 | – |
| R3 → see B | plain-bg ablation on GPU (OOM-killed, replaced by B) | – | – |
| R2 (GPU, w32) | full recipe **without distractors** | 0.9816 | – |
| R4 (GPU, w32) | focal loss instead of BCE | 0.9838 @ep23 vs 0.9856 BCE @ep22 (stopped) | – |
| R5 (GPU, w16) | narrower model | 0.9884 | – |
| **R1** (GPU, w32) | full recipe + frog/big-arch distractors | **0.9894** | 0.9576 / 0.9573 |
| R1 + hour prior | × activity(hour)^0.5 from train hours | – | 0.9463 / 0.9495 ✗ |
| **PF1** | + round-2 pseudo-label test backgrounds (all clips) | 0.9891* | 0.9626 / 0.9607 |
| **final ensemble** | R1 + F1 + F2 + PF1 + PF2, TTA×4 | – | 0.9636 / 0.9627 |

\* full-data models also train on the held-out clips, so their val AUC is optimistic.

## What mattered most
1. Synthetic soundscape mixtures with test-derived backgrounds + distractors (val 0.86 → 0.94 at equal budget; test 0.73 → 0.957).
2. Looking at test false positives and adding the missing distractor types (frog chorus, big arches).
3. Pseudo-labelled real test backgrounds (round 2): +0.003–0.005 on the test.
4. Not trusting the train hour prior: it looked compelling but cost ~0.01 AUC on the real test.
5. Ensembling seeds / rounds + time-shift TTA.

## Status / hand-off notes (for v2)
* Remote jobs on aorus-ts are stopped. Checkpoints (`R1_full, F1, F2, PF1, PF2 .pt/.json`, ~5 MB each) are in the local scratch
  cache (`CACHE_3A`, default `/tmp/claude-1000/-home-minh-Desktop-olympiad-ai/1f52811c-e600-457a-bc8e-50e6365e680a/scratchpad/cache`);
  `python predict.py R1_full F1 F2 PF1 PF2 --tta 4` with that cache reproduces the CSVs exactly (verified, max diff 1e-6).
* The full `solution.py` run was not re-executed end-to-end on the local CPU (≈2.5 h there); each step was run individually
  (training on aorus-ts, inference locally). score.py was called 4 times: R1, R1 + hour prior, PF1, final ensemble.
* Unfinished: ablation R3 (plain background on GPU) died of OOM, R6 (no SpecAugment) never ran, R4 (focal loss) was stopped at epoch 23.

**Best ideas for further gains**
1. Iterate the pseudo-label rounds (round 3 from the 5-model ensemble; sharper/thresholded targets; higher `p_test_bg`) — round 2 gave +0.003–0.005.
2. Fix the silphe/frog confusion: models still flag silphe in ~39 % of test clips. Try a better frog-chorus distractor (match the test pulse rate and
   harmonic spacing), or make its pseudo-labels for silphe at night less trusted.
3. Bigger or pretrained backbone (torchvision ResNet18/EfficientNet on 3-channel log-mel) now that a GPU is available; longer training (80+ epochs).
4. Mix in the waveform domain (real phase interaction) or train at full 8 kHz STFT resolution; add time-stretch and per-band gain augmentation.
5. Per-species calibration does not change AUC, but ranking ensembles (rank-average) and more seeds are cheap wins. The hour/site prior hurt once, so leave it out unless a much softer prior (β ≈ 0.1) is validated.
