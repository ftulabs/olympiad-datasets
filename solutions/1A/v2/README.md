# 1A - VN30 next-day return, v2: recovering the data-generating process

| | Own CV (2023 / 2024 forward folds, mean) | Public RMSE | Private RMSE |
|---|---|---|---|
| v1 (wide-and-deep MLP, 8 seeds) | 1.1336 | 1.1310 | 1.1421 |
| **v2 (structural model + 10% v1 NN)** | **1.1094** | **1.1094** | **1.1274** |
| Approximate ceiling (task statement) | - | - | about 1.12 |

v2 cuts the private RMSE by **0.0147** (from 1.1421 to 1.1274) and closes about two-thirds of the gap between v1 and the ceiling.

Files: `solution.py` runs end-to-end in about 5 minutes on 2 CPU threads; `python solution.py --cv` reproduces the CV. `solution.ipynb` holds the same pipeline in the sections Data, Model, Training & Inference and Predict, with no submit cell. The folder also has `public_submission.csv` and `private_submission.csv`.

Compute: everything ran on the local CPU (2 threads, under 2 GB of RAM). No GPU or remote host was needed, because the final model has about 90 parameters.

## Validation
- **Main scheme:** the same expanding forward folds as v1. Train on years before 2023 and validate on 2023, then train on years before 2024 and validate on 2024. The test sets lie in 2025.
- **Second check: leave-one-year-out (LOYO).** I first checked that the process does not change over time: the feature distributions match across 2021-2025, there is no autocorrelation, and the residuals have no date-level component. Each year is then an exchangeable fold, and LOYO trains on 3 years, which is closer to the final refit on 4.
- **score.py:** it was used only after the final model had been chosen on CV, plus one ablation run. See the call log below.

## 1. Data perspective: reverse-engineering the generator
The data is synthetic. The EDA rebuilt the generator one term at a time. For each step I conditioned on regimes, checked binned partial residuals, and probed the residuals with a GBM plus permutation importance.

1. **Inputs are generated from z-scores.** The spread of `ret_1d`, `ret_5d` and `ret_20d` grows like `vol`, `2.3·vol` and `4.5·vol` (that is, sqrt(1), sqrt(5) and sqrt(20) times vol). `rsi_14` does not depend on vol, and RSI is about 50 + 3·ret_5d (R² 0.69). So `ret_5d` carries no information beyond RSI: its fitted weight is 0.
2. **Volume regimes with sharp thresholds.** Regressing the target on `ret_1d` within volume-ratio bins gives a slope of **-0.21 below 1.20** and **+0.11 above 1.20**. The switch is a step. A threshold grid (1.15, 1.18, 1.19, **1.20**, 1.21, 1.22, 1.25) put the minimum exactly at 1.20.
3. **The return term is volatility-normalised.** The model learns `x = ret_1d / vol^k` with k = 0.95-0.97. In the 2D tables, the partial residual is linear in x within each regime.
4. **A second threshold at volume_ratio 2.0 ("breakout").** The fine 2D table of partial residuals showed a **jump of about ±0.45 at ret_1d = 0**, but only where volume_ratio ≥ 2.0. Rows at 1.8-2.0 show no jump, while rows at 2.0-2.2 do. So there are three regimes: reversal (vr < 1.2), momentum (1.2 ≤ vr < 2.0), and momentum plus a sign step (vr ≥ 2.0). The step is fitted as `tanh(x/0.1)`.
5. **The ret_1d sensitivity is scaled per ticker.** Letting each coefficient vary by ticker, one term at a time, moved CV only for the ret_1d terms. The low-volume and high-volume slopes, fitted separately per ticker, correlate at **-0.91** across tickers. That points to one multiplier g_ticker, which ranges from 0.15 to 1.58. The best fit applies g to the reversal and momentum slopes but not to the breakout step.
6. **The remaining terms are simple shapes,** read from spline fits and then made parametric:
   - `-0.90·tanh((rsi-50)/12)`
   - `+0.54·tanh(fnb/44)`
   - `+0.069·ret_20d / vol^0.50`: the exponent was learned as 0.4998. The raw form and the /vol form are both worse.
   - market return times a **sector** beta: banking 0.35, consumer -0.05, energy 0.24, materials 0.44, real estate -0.28, technology 0.12. Per-ticker betas gain nothing.
   - a **Monday** effect of -0.26. The other weekdays are 0.
   - a **ticker alpha** with sd 0.33.
   - `usd_vnd_change`, `ret_5d` and raw `volatility_20d` have no effect. The NaN flag on foreign flow has weight 0, so filling it with 0 is correct.
7. **The noise is heavy-tailed and heteroscedastic.** It fits a Student-t with nu = 4.0 and scale `0.56·vol^0.48`, roughly sqrt(vol). The residuals show no date-level shock: the spread of the date-mean residual is 0.204 against 0.200 if rows were independent, and the mean pairwise correlation between tickers is 0.002. They are also uncorrelated across days and across tickers.
8. **Checking for missed structure.** On the final structural model, the residual GBM and a residual MLP each recover less than 0.0003 RMSE. The in-sample against out-of-sample gap per year is about 0.002-0.003. That gap is the estimation error, so almost all of what remains is irreducible noise.

## 2. Model perspective
- **Final structural model:** an `nn.Module` with about 90 parameters (30 alphas, 30 ticker multipliers g, 6 betas and the shape parameters). It is fitted by full-batch LBFGS in float64, using the **Student-t negative log-likelihood with a learned scale that depends on vol**. Matching the likelihood to the heavy-tailed noise gives more efficient estimates than MSE (about -0.0008 on CV) or Huber.
- **Three variants of the breakout term, averaged:**
  - the step has its own unscaled slope;
  - the mid-regime slope (scaled by g) continues into the top regime, plus a step and a linear term;
  - the same as the second, without the linear term.
- **Diverse member:** the v1 wide-and-deep MLP (8 seeds), blended at weight 0.10, which was chosen on CV. It improves both folds by 0.0002-0.0004.

## 3. Training and inference perspective
- LBFGS fits the structural model to convergence, so there are no epochs to tune. Bootstrap bagging (10 bags) did not help (1.1101 against 1.1099), and neither did ridge shrinkage of the alphas or of g (unchanged at λ = 0.5 and 2). There are about 1,000 rows per ticker, so the estimates are already precise.
- **Final refit on all 2021-2024 data.** The NN member uses v1's fixed schedule: 40 epochs of OneCycle at learning rate 3e-3 with Huber loss.
- TTA and post-processing do not apply: the rows are independent, and the model already gives the conditional mean.

## Ablation log (own validation)
All rows below the NN-only rows are linear ridge models up to and including the spline step, and PyTorch parametric models after it. CV is the mean of the forward folds 2023 and 2024. LOYO is the mean of the four leave-one-year-out folds.

| Step | Model / change | 2023 | 2024 | CV mean | LOYO |
|---|---|---|---|---|---|
| v1 | wide-and-deep MLP, 3 seeds | 1.1415 | 1.1257 | 1.1336 | |
| s0 | additive splines + ticker + dow (no interactions) | 1.2268 | 1.2465 | 1.2366 | |
| s1 | + market x sector, ret_1d x 1{vr≥1.2} (linear) | 1.1449 | 1.1244 | 1.1346 | |
| s1 | threshold 1.15 / 1.18 / 1.22 / 1.25 | | | 1.1414 / 1.1379 / 1.1364 / 1.1422 | |
| s2 | ret_1d splines per regime | 1.1391 | 1.1223 | 1.1307 | |
| s3 | **ret_1d / vol** splines per regime | 1.1353 | 1.1170 | 1.1262 | |
| s3 | + ret_20d/vol, ret_5d/vol variants | | | 1.1251 to 1.1287 | |
| p1 | parametric (tanh RSI/flow, linear r1v per regime), MSE | 1.1392 | 1.1177 | 1.1284 | |
| p1 | + tanh-saturating high-regime term | 1.1365 | 1.1159 | 1.1262 | |
| p2 | + ret_20d / vol^k (k learned to 0.50) | 1.1348 | 1.1151 | 1.1249 | |
| p2 | Huber / Gaussian-heteroscedastic / **Student-t** NLL | | | 1.1245 / 1.1250 / **1.1245** | |
| p3 | per-ticker coefficient, one at a time: rsi / flow / ret_20d / Monday | | | 1.1249 / 1.1249 / 1.1253 / 1.1249 | |
| p3 | per-ticker market beta | | | 1.1240 | |
| p3 | per-ticker **low-regime ret_1d slope** | 1.1260 | 1.1068 | **1.1164** | |
| p3 | per-ticker low and high slopes | 1.1234 | 1.1062 | 1.1148 | |
| p3 | one multiplier g_ticker on the ret_1d block | 1.1232 | 1.1069 | 1.1151 | |
| p4 | **+ breakout regime vr ≥ 2.0 (sign step)** | 1.1196 | 1.1032 | 1.1114 | 1.0964 |
| p4 | second threshold 1.9 / 2.1 | | | 1.1128 / 1.1120 | |
| p4 | first threshold 1.19 / 1.21 | | | 1.1112 / 1.1125 | 1.0970 / 1.0978 |
| p4 | MSE instead of Student-t | 1.1208 | 1.1039 | 1.1123 | 1.0970 |
| p5 | **g not applied to the breakout step** | 1.1193 | 1.1005 | 1.1099 | 1.0952 |
| p5 | mid slope continues into top (variant B) | 1.1187 | 1.1016 | 1.1102 | 1.0951 |
| p5 | + per-ticker beta (L2) | 1.1196 | 1.1028 | 1.1112 | 1.0964 |
| p5 | shrink g or alpha (L2 0.5, 2) | | | 1.1099 | 1.0951-1.0952 |
| p5 | bootstrap bagging x10 | 1.1195 | 1.1007 | 1.1101 | |
| p5 | + residual MLP on top (3 configs) | 1.1191 | 1.1002 | 1.1097 | |
| p6 | **average of 3 breakout variants** | 1.1185 | 1.1008 | **1.1096** | 1.0948 |
| p6 | + v1 NN at weight 0.05 / **0.10** / 0.20 | 1.1183 (at 0.10) | 1.1004 (at 0.10) | 1.1094 / **1.1094** / 1.1097 | |

**Final model (selected on CV before any scoring):** p6 with the NN at weight 0.10, refit on all of 2021-2024.

## score.py calls (2 of the 8 allowed)
| # | Submission | Public | Private |
|---|---|---|---|
| 1 | **final v2** (structural average + 10% NN) | **1.1094** | **1.1274** |
| 2 | ablation: structural average only, no NN | 1.1091 | 1.1280 |

Both runs agree with CV: the NN blend is worth about ±0.0005. The selection was not changed after scoring.

## Comparison with v1 and the ceiling
- On private, v2 scores 1.1274 against v1's 1.1421 (-0.0147). On public, 1.1094 against 1.1310 (-0.0216). On CV, 1.1094 against 1.1336 (-0.024).
- The task gives an approximate private ceiling of about 1.12. v2 is about 0.007 above it. My own model implies noise floors of 1.0935 (public) and 1.0906 (private). These come from the fitted Student-t: sigma = 0.56·vol^0.48 and nu = 4.0, which gives a variance of 2·sigma². They are rough, because a t-distribution with nu = 4 has an extremely heavy-tailed sample variance.

## What limits further gains
1. **Irreducible noise dominates.** The residuals are independent Student-t noise with no date, ticker or cross-sectional structure left. A residual GBM or MLP recovers less than 0.0003.
2. **Estimation error is about 0.002-0.003 RMSE.** This is the gap between in-sample and out-of-sample error in each year. It comes mostly from the 30 alphas and 30 ticker multipliers, which are estimated from about 1,000 noisy rows each. Shrinkage and bagging did not reduce it, so the error is close to its statistical minimum for this sample size. The only remedy would be more labelled data.
3. **Possible small misspecification** in the exact shape of the breakout step or of the saturating functions (tanh against something else). The three variants of the breakout term differ by under 0.0005, so any remaining error here is tiny.
4. **The noise is heavy-tailed** (nu about 4), so a single half-year of test data has an RMSE spread of about ±0.01. The public/private difference (1.109 against 1.127) is mostly that spread, not model error.
