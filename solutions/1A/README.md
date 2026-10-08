# 1A - VN30 next-day return (regression, RMSE)

**Final scores (score.py, run once on the final files):** public **1.1310**, private **1.1421** -> **Gold** (gold threshold 1.15; baseline about 1.244 private).

Final model: a PyTorch wide-and-deep MLP (residual block, ticker/sector/day embeddings, per-ticker bias) on engineered features. It is trained with Huber loss and an 8-seed ensemble, refit on all of 2021-2024.
Files: `solution.py` (end-to-end, about 5 min on 1 CPU thread), `solution.ipynb` (the same pipeline in sections), `public_submission.csv`, `private_submission.csv`.

## Validation scheme
The test set lies in the future (2025), so all model selection used **two expanding time folds**: train on rows before 2023 and validate on 2023, then train on rows before 2024 and validate on 2024. The number reported is the mean RMSE over the two folds. Predicting zero scores about 1.47 on 2024. Final settings were chosen by this CV only. score.py was run once, on the final files.

## 1. Data perspective (most of the effort)
EDA findings:
- 31,260 rows: 30 tickers x 1,042 days, 6 sectors. `foreign_net_buy_bn` is 4.9% NaN. The NaNs look random: they are spread evenly across tickers and the target is the same with or without them. They are filled with 0 and marked by a flag column.
- **Marginal signals:** `rsi_14` (corr -0.24, peaks around decile 2) and `ret_5d` (-0.13) show reversal. `ret_20d` (+0.14) and `foreign_net_buy_bn` (+0.21) show momentum. Mondays average -0.29 against about 0 on other days.
- **Strong, stable ticker effect:** per-ticker mean returns spread with sd 0.33, and ticker means correlate about 0.95 from year to year. Adding a ticker id took the GBM probe from 1.225 to 1.152, the biggest single gain.
- **Hidden interactions:** `volume_ratio` has about 0 marginal correlation, but dropping it costs 0.065 RMSE. It flips the sign of the `ret_1d` slope (-0.23 at low volume, +0.14 at high volume: reversal versus momentum), and of the `market_ret_1d` slope too. Market beta also depends on sector (real estate -0.34, materials +0.41).
- Noise is heteroscedastic: the target sd rises from 1.24 to 1.74 across `volatility_20d` deciles. The target is heavy-tailed (kurtosis 3.4) and so are the inputs (`volume_ratio` goes up to 29).
- Rows carry **no serial information.** Lag-1 autocorrelation of every feature is about 0, and lagged features do not correlate with the target. Lag and rolling features were therefore skipped. `usd_vnd_change` is noise and was dropped.
- Date-level cross-sectional features (demeaned or date-mean versions) did not help the probe (1.159 and 1.154 against 1.152).

Transforms: log of `volume_ratio`; returns divided by volatility (`r1v`, `r5v`, `r20v`); the interactions `ret_1d*log(vr)` and `mkt*log(vr)` plus a high-volume flag; inputs clipped at the train 0.5% and 99.5% quantiles, then standardised (fitted on the training part only).

## 2. Model perspective
- An embedding for each categorical: ticker (8 dims), sector (4) and day of week (3).
- A small residual MLP: hidden 128, 2 linear layers, LayerNorm, SiLU, dropout 0.4. It runs alongside a **wide linear path** and a **per-ticker bias**.
- Huber loss (delta 1.0) instead of MSE, because the target is heavy-tailed.
- The data has a low signal-to-noise ratio, so capacity hurts. A 4-layer, 256-wide net without strong dropout overfit badly (1.201).

## 3. Training and inference perspective
- AdamW (weight decay 1e-4) with a OneCycle schedule (max LR 3e-3, 20% warm-up), 40 epochs, batch 512.
- The epoch count is fixed from the CV curves instead of early stopping, so the final model can be refit on **all of 2021-2024**.
- An 8-seed ensemble (averaged predictions). Going from 3 to 6 seeds was worth about 0.001.
- TTA does not apply to tabular rows that do not depend on each other.

## Ablation log (validation RMSE, mean of the 2023 and 2024 folds)
| # | Change | CV RMSE |
|---|---|---|
| probe | GBM (HistGB), raw 10 features | 1.2247 |
| probe | GBM + ticker id | 1.1517 |
| probe | GBM + ticker + vol-normalised returns | 1.1474 |
| probe | GBM + ticker, drop `volume_ratio` | 1.2170 |
| probe | GBM + ticker, drop `usd_vnd_change` | 1.1493 |
| NN-0 | MLP 3x128, drop 0.1, MSE, embeddings + FE, 30 ep, 1 seed | 1.1541 |
| | no feature engineering (raw + NaN flag) | 1.1576 |
| | no quantile clipping | 1.1551 |
| | drop `usd_vnd_change` | 1.1513 |
| | Huber (delta 2) instead of MSE | 1.1479 |
| | 4x256 (bigger) | 1.2008 |
| | 2x64 (smaller) | 1.1460 |
| | dropout 0.3 | 1.1457 |
| | no wide path | 1.1555 |
| | ReLU instead of SiLU | 1.1632 |
| | batch 256 / batch 2048 | 1.1627 / 1.1507 |
| NN-2 | drop usd + Huber + 2x64 drop 0.3, 20 ep, 3 seeds | 1.1443 |
| | MSE | 1.1462 |
| | Huber delta 1.0 | 1.1431 |
| | sample weights 1/vol, 1/vol^2 | 1.1438 / 1.1455 |
| | no FE (in this config) | 1.1523 |
| | hidden 128, dropout 0.4 | 1.1408 |
| | 30 epochs | 1.1397 |
| NN-3 | 2x128 drop 0.4, Huber delta 1, 30 ep, 3 seeds | 1.1367 |
| | 40 epochs | 1.1349 |
| | 3 layers / 256 wide drop 0.5 | 1.1355 / 1.1364 |
| | ticker embedding 4 instead of 8 | 1.1390 |
| NN-4 | 40 ep + LR 3e-3 (**selected**) | **1.1336** |
| | 60 ep at LR 2e-3 | 1.1336 (same, slower) |

**Final (selected on CV):** NN-4 config, refit on 2021-2024, 8 seeds. Public **1.1310** / private **1.1421**, Gold tier.

## What mattered most
1. Ticker identity (embedding plus bias): about -0.07 RMSE.
2. Keeping and engineering the volume-regime interactions (`ret_1d` and `market_ret_1d` times log volume ratio) and the vol-normalised returns: about -0.008 on the NN, and essential on the GBM probe.
3. Strong regularisation and small capacity (dropout 0.4, 2 layers, wide path): about -0.01 compared with a deeper or less regularised net.
4. Huber loss (delta 1): about -0.004.
5. Seed ensembling and a full refit with a fixed number of epochs: a small but free gain.
