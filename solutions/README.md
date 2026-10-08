# Olympiad AI warm-up · reference solutions (1A – 4B)

Each task has two solutions, built like a competitor would: only the dataset and the baseline notebook, with the
final model chosen on the solver's own validation. The hidden labels were used only to score the final files.

- `<task>/` — **v1**: first pass that beats the Gold tier.
- `<task>/v2/` — **v2**: "push to the limit" pass that starts from v1.

Each folder holds `solution.py` (end-to-end), `solution.ipynb` (sample notebook: Data / Model / Training & Inference /
Predict) and `README.md` (the log: findings, the full ablation table with validation numbers, every scoring call).
It also holds the two submission files.

Every solution follows the same three-perspective method: **Data** (50–60% of the effort), **Model** (~30%),
**Training & Inference** (10–20%).

## Results (private score; ranking uses private)

| Task | Metric | Baseline | v1 | v2 | Approx. ceiling | Gold line |
|---|---|---|---|---|---|---|
| 1A VN30 next-day return | RMSE ↓ | 1.244 | 1.142 | **1.127** | 1.12 | ≤ 1.15 |
| 1B E-wallet fraud | F1 ↑ | 0.32 | **0.588** | 0.583 * | ~0.58–0.59 | ≥ 0.56 |
| 2A Traffic detection | mAP@0.5 ↑ | 0.227 | 0.872 | **0.950** | 1.0 | ≥ 0.60 |
| 2B Rice-leaf multimodal | macro-F1 ↑ | 0.611 | 0.921 | **0.946** | unknown | ≥ 0.84 |
| 3A Forest bird calls | macro ROC-AUC ↑ | 0.731 | 0.963 | **0.972** | 0.979 | ≥ 0.94 |
| 3B Molecule binding | mAP ↑ | 0.270 | 0.427 | **0.446** | 0.575 | ≥ 0.42 |
| 4A Comment moderation | column-avg AUC ↑ | 0.683 | 0.927 | **0.932** | 0.946 | ≥ 0.91 |
| 4B Tết demand forecast | WAPE ↓ | 0.616 | 0.419 | **0.369** | 0.342 | ≤ 0.42 |

\* 1B: v2 is the better-founded model. It won on 95% of resampled validation sets. With only ~470 frauds per test
split, F1 moves by about ±0.016, so v1's private score was partly luck. See `1B/v2/README.md`.

## Common themes

1. **Build a validation that looks like the test.** This mattered most in every task:
   - 1A: forward time folds.
   - 4B: Tết-aligned backtests.
   - 3B: whole building blocks and a scaffold held out.
   - 4A: rules held out.
   - 3A: synthetic soundscapes.
   - 2A: a darkened, zoomed-out copy of the hold-out.
2. **Measure the train → test shift and design around it.** Examples: blur augmentation (2B), mosaic zoom-out (2A), mixing clips into
   test-like noise (3A), "days to Tết" instead of month/day (4B).
3. **Use the structure the baseline ignored.** Ticker identity, bond types, the rule text, round amounts, scaffold +
   building blocks, and labelled data hidden in the inputs (4A's example columns, pseudo-labels in 2A/3A).
4. **Small, well-specified models near the noise floor.** Pretrained torchvision backbones where images or audio carry the
   signal (allowed by the rules).
5. **Train for the metric, then ensemble.** Huber / Student-t, price-weighted Poisson, F1 threshold chosen out-of-fold,
   near-zero score threshold for mAP; seed and fold ensembles; test-time augmentation.

## Why some tricks failed (each README logs its rejected ideas)

| Root cause | Examples |
|---|---|
| Fixing a problem the data doesn't have | focal loss / class weights (1B), per-class thresholds (2B), pseudo-labels with no train-test shift (1B) |
| Train-only patterns that don't hold in test | the owl's night-only `hour` prior (3A), "same day last season" with censored peaks (4B) |
| No new information | extra seeds or models with prediction correlation 0.996; pseudo-labels that double-count other modalities (2B) or repeat the model's own errors (3B, 4A) |
| Too much capacity near the noise floor | wide MLPs on 1A, extra stackers on 4A |
| Gain smaller than metric noise | 1B v1 vs v2 on 470 positives — use paired comparisons and bootstraps on out-of-fold predictions |
| Hardware | AMP slower on a GTX 1650; large backbones out of memory on 4 GB GPUs |

Before using any trick, ask four questions:
1. What problem does it solve, and do I see that problem in my data?
2. Is the pattern real for the test, or an artefact of how train was collected?
3. Does it add information, or only capacity?
4. Is the gain bigger than my validation noise?
