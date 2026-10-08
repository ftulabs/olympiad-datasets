# 1B · E-wallet fraud — solution write-up

**Result (score.py, one scoring call):** public F1 = **0.5751**, private F1 = **0.5884** → **Gold** (Gold ≥ 0.56; baseline ≈ 0.32).
Model selected only on own 5-fold stratified out-of-fold (OOF) F1: **0.570** (AP 0.614).

Files: `solution.py` (end-to-end, ~4.5 min CPU single thread, ~0.7 GB RSS), `solution.ipynb` (same pipeline, executed, reproduces the CSVs bit-for-bit), `public_submission.csv`, `private_submission.csv`.

## 1. Data (most of the effort)
EDA on train (59,854 rows, 4.66 % fraud, no missing values, ids randomly split across train/public/private → no time drift):

| Finding | Fraud rate |
|---|---|
| log(amount / avg_amount_30d) < -3 (tiny vs. usual spend, "card testing") | 97 % |
| log-ratio > 2 (much larger than usual) | 31 % |
| amount < 20,000 VND | 27 % (≤ 5k: 72 %) |
| amount a multiple of 50,000 VND (round amounts) | 10-11 % vs 4 % otherwise; multiples of 10k that are not 50k show no effect |
| account_age_days ≤ 30 (sharp step at 30) | ~15 % vs 4 % |
| tx_count_24h ≥ 5 / failed_pin_24h ≥ 2 / new_device | 6-33 % / 15-17 % / 17.5 % |
| night hours 0-5 | 7-9 % (new_device at night: 42 %) |
| distance > 100 km (flat below) | 10-12 % |
| web channel within every category (gift_card/web 29 %, utilities ≈ 0.4 %) | ~2x |

Outliers: account ages up to 69,407 days (190 years) → capped at 3,650; distances up to 2,500 km → log1p and cap 1,000.
Transforms/features (all row-wise, no fitted statistics → no leakage): log amount, log avg, log ratio split into positive/negative parts (U-shape), flags `tiny_amt`, `round_50k`, `round_10k`, `young`, `night`, `tx_hi`, `far`; cyclic hour; log age; clipped counts; category, channel and category×channel as embeddings. Numeric features are standardised with the training-fold mean/std.
Imbalance: kept natural class balance with plain BCE and handled it with the decision threshold (pos_weight=3 and focal loss were no better).
Validation: 5-fold StratifiedKFold (seed 0), metric = best-threshold F1 on concatenated OOF predictions (plus AP as a threshold-free check).

## 2. Model
`FraudNet`: [standardised numerics ‖ emb(cat,4) ‖ emb(channel,2) ‖ emb(cat×channel,4)] → Linear(64) → 1 pre-activation residual block (BN → SiLU → Linear → SiLU → Dropout 0.2 → Linear, + skip) → BN/SiLU/Linear(1). Small on purpose: the signal is low-dimensional and noisy; wider/deeper nets overfit.

## 3. Training & inference
AdamW (wd 1e-4), OneCycle LR max 2e-3, batch 512, **8 epochs** (no early stopping on the validation fold, so the OOF estimate is honest). 5 folds × 3 seeds = 15 models. The threshold is chosen on OOF predictions (F1 curve smoothed over neighbouring cut points; chosen 0.294 ≈ top 4.6 % flagged) and the test probabilities are the mean of the 15 fold models, so they are calibrated like the OOF predictions the threshold was tuned on.

## Ablation log (5-fold OOF, 1 seed unless noted)
| # | Change | OOF AP | OOF F1 |
|---|---|---|---|
| 0 | Baseline-like: raw numerics (+one-hot cats), 1×32 MLP, 20 ep, OOF-tuned threshold | 0.499 | 0.474 |
| – | (reference only, not submitted) sklearn HistGB raw → with my features | 0.570 → 0.602 | 0.528 → 0.556 |
| 1 | Residual MLP 128, depth 2, 30 epochs, + features | 0.561 | 0.530 |
| 2 | same, 12 epochs (overfitting was the problem) | 0.604 | 0.562 |
| 3 | 12 ep, no features (raw numerics + embeddings) | 0.527 | 0.491 |
| 4 | 12 ep, one-hot instead of embeddings | 0.607 | 0.566 |
| 5 | 12 ep, dropout 0.4 / wd 1e-3 | 0.607 | 0.565 |
| 6 | hidden 64, depth 1, 12 ep | 0.610 | 0.568 |
| 7 | **hidden 64, depth 1, 8 ep** (chosen) | 0.612 | 0.571 |
| 8 | 7 minus round-amount flags | 0.573 | 0.534 |
| 9 | 7 minus threshold flags (tiny/young/far/night/tx_hi) | 0.583 | 0.538 |
| 10 | 7 minus ratio_pos/ratio_neg split | 0.604 | 0.563 |
| 11 | 6 with pos_weight=3 | 0.607 | 0.565 |
| 12 | 6 with focal loss (γ=1) | 0.609 | 0.568 |
| 13 | 7 with depth 0 / depth 2 / hidden 128 (6 ep) | 0.603 / 0.610 / 0.607 | 0.559 / 0.567 / 0.565 |
| 14 | 7 with constant LR 1e-3 (no OneCycle) | 0.605 | 0.565 |
| 15 | 7 with 3 seeds | 0.613 | 0.571 |
| **Final** | 7, 5 folds × 3 seeds, smoothed OOF threshold | **0.614** | **0.570** |

Differences below ~0.005 F1 are within fold/seed noise (≈ 2,800 positives).

## What mattered most
1. **Feature engineering** (+0.08 F1 over raw inputs): round-amount flag (+0.037) and step flags at the observed breakpoints (+0.033), U-shaped amount/avg ratio.
2. **Short training / small network**: 30 → 8 epochs and 128×2 → 64×1 gave +0.04 F1.
3. **F1-optimal threshold chosen on OOF** instead of a fixed 0.3, with test probabilities from the same fold models.
4. Seed/fold ensembling: small but consistent AP gain, more stable threshold.

Scoring: score.py was called once, on the final files only; nothing was tuned on public/private.
