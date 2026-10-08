# 1B · E-wallet fraud: v2 (pushing toward the ceiling)

**Final v2 (score.py, 1 call in total): public F1 = 0.5747, private F1 = 0.5825 (Gold ≥ 0.56).**
v1 scored public 0.5751 / private 0.5884.
All selection used only my own validation, which is 3 × repeated stratified 5-fold out-of-fold (OOF) predictions:

| | OOF AP | OOF log-loss | OOF F1 (smoothed-threshold) |
|---|---|---|---|
| v1 (residual MLP, 5 folds × 3 seeds) | 0.614 | ≈0.1007 | 0.570 |
| **v2 (spline-GAM + generator-matched interactions)** | **0.633** | **0.0978** | **0.582** |

Files: `solution.py` (end-to-end, ~45 s on 2 CPU threads, 0.74 GB RSS, auto-selects CUDA if present), `solution.ipynb` (same pipeline split into Data / Model / Training & Inference / Predict, executed, reproduces the CSVs bit-for-bit), `public_submission.csv`, `private_submission.csv`, `oof_and_test_probs.npz` (OOF and test probabilities plus threshold, 0.4 MB).

## Where the ceiling is, and why v2's test score did not move
* **The model is at its own Bayes limit.** If labels are drawn as Bernoulli(p_v2), the expected F1 on the OOF rows is 0.5820 ± 0.007, and the observed OOF F1 is 0.5819. So the model is calibrated, and the screening, the residual MLP and the mixture model all agree that no learnable structure is left (see the ablations). The model-implied F1 on the test splits is **public 0.589 ± 0.017, private 0.579 ± 0.015**. The actual scores of 0.575 and 0.5825 both fall within 1 SD of that. I estimate the practical ceiling at **≈ 0.58–0.59**, with a test-sampling SD of **±0.016** per split of 10k rows (about 470 positives).
* **v1 vs v2 on test is dominated by noise.** I simulated 2,000 random subsets of 10k rows from the OOF rows. On them, the paired difference F1(v2) − F1(v1-like) was +0.016 ± 0.009, and v2 won in 95 % of subsets. The realised private difference of −0.006 is an unlucky draw, and v1's private 0.588 was itself +0.018 above its OOF score. Adversarial validation of train against public and private gave AUC 0.497 / 0.501, so there is no distribution shift that a different validation design could have fixed. I did **not** use score.py to choose between variants.
* An oracle threshold on each test split would add only about +0.007 (simulated `oracle` 0.591 against the OOF-threshold rule 0.584).

## 1. Data: the data-generating process
EDA for v1 is in `../README.md`. New findings for v2:
1. **The generator is logistic with explicit pairwise terms.** An additive GBDT (depth 1) is clearly worse than a depth-2 one (OOF log-loss 0.1056 → 0.1014). I screened the residuals of an additive spline-logistic model with a score test over about 300 candidate products. The real interactions were: round-50k × category (round amounts are *normal* for p2p transfers: 29 % of legit p2p rows are multiples of 50k, against 5-6 % elsewhere); new_device × amount ratio (new device and ratio 1-2: 62 % fraud against 41 % additive-expected); far (>100 km) × channel (far+app 3.5 %, far+qr 22 %, far+web 17 %); night × new_device (42 % against 25 %); night × p2p. After these were added, a re-screen showed max |z| ≈ 3 over 300 tests, which is what pure noise gives.
2. **Card-testing mode.** Fraud amounts below 20k VND are spread ~uniformly over 1k–20k (linear-uniform, so the density rises in log space), regardless of the user's average. **No** row with amount ≥ 20k and log-ratio < −2.5 is fraud, while tiny amounts with ratio < −3 are 95–100 % fraud. Feature `tiny × ratio_neg` gave the single largest new gain (Δ log-loss −0.0003, reproduced on a second CV split).
3. **Rounding.** Only multiples of 50k carry signal. Multiples of 1k, 5k or 10k that are not multiples of 50k sit at the base rate (3.6 / 2.7 / 3.1 % against 3.7 %), so v1's `round_10k` was dropped along with the 1k flags. Round amounts ≥ 1M and round amounts at night get their own terms.
4. **Thresholds made exact.** Account-age step between 29 and 30 days (16.9 % → 3.8 %), so `young = age < 30` (v1 used ≤ 30). Distance step at 100 km. tx_count rises from 5 upward (`tx_x = max(tx−4, 0)`). failed_pin ≥ 2.
5. **Checked and rejected.** I looked for user identity through (avg_amount_30d, account_age) duplicates: there are 750 collision pairs, but their fraud rate equals the base rate, so there is no user leakage. id order carries no signal. Train and test are identically distributed (adversarial AUC 0.50). Pseudo-labelling was not used: for a calibrated, well-specified discriminative model the unlabelled test rows carry no information about p(y|x), and with no shift there is nothing to adapt to.

## 2. Model
`SplineGAM` (PyTorch): logit = w · [standardised features, ReLU(x_c − knot_j) for 8 quantile knots on 8 continuous features] + b[category × channel]. It has 34 input columns and 98 weights. A ridge penalty applies to the spline hinge weights (3e-3). The category × channel bias is almost unpenalised (1e-6). **That was a real bug in my first GAM:** a 1e-3 penalty on that bias pulled utilities (0.4 % fraud) toward the mean and over-predicted it by 2× (z = −2.7 in the calibration check). Fixing it improved OOF log-loss by 0.0009 and AP by +0.004.
Other families I tried (all PyTorch): the v1 residual MLP (with v2 features), a wider/deeper MLP, a wide & deep MLP, a GAM-offset residual MLP (boosting the GAM with an MLP), a log-sum-exp mixture of 2-3 additive "fraud scenarios", and focal and soft-F1 losses. None beat the GAM out-of-fold, and logit-blends of GAM with MLP only got worse as MLP weight grew. A misspecified flexible model loses to a correctly specified one with 2,800 positives.

## 3. Training & inference
L-BFGS full batch (convex, so deterministic: seed ensembles are pointless and were replaced by CV repeats). Validation is 3 repeats × 5 stratified folds; the OOF probabilities are averaged over repeats. The threshold maximises the 9-point smoothed OOF F1 curve (0.327, which flags 3.75 % of rows). The plateau runs from 0.30 to 0.36 (F1 0.580–0.582). In threshold-rule simulations on held-out chunks of 10k rows, the OOF threshold, the expected-F1 plug-in threshold, a fixed 0.30 and matching the OOF flag rate all scored within 0.001 of each other. Test probabilities are 0.5 × the mean of the 15 fold models plus 0.5 × a refit on all 59,854 labelled rows.

## Ablation log (own validation; 5-fold OOF on the same split unless noted; LL = log-loss)
Differences below ≈ 0.0002 LL or ≈ 0.004 F1 are within noise, so LL and AP are the primary criteria.

| # | Model / change | AP | LL | F1 |
|---|---|---|---|---|
| ref | HistGB depth 1 (additive) / depth 2 / deep, raw + few features (reference only) | 0.582 / 0.610 / 0.601 | 0.1056 / 0.1014 / 0.1026 | 0.543 / 0.562 / 0.554 |
| ref | HistGB depth 2, v1 features → v2 features (reference only) | 0.611 → 0.616 | 0.1015 → 0.1008 | 0.565 → 0.570 |
| M0 | v1 MLP, v1 features (1 seed) | 0.612 | 0.1017 | 0.566 |
| M1 | v1 MLP + v2 interaction features | 0.616 | 0.1007 | 0.572 |
| M1a | M1 − interaction block | 0.613 | 0.1014 | 0.566 |
| M1b | M1 − tx_x / pin2 | 0.613 | 0.1014 | 0.571 |
| M1c | M1 with 6 / 12 epochs, SWA (10 ep), h128×2, wide&deep | 0.615–0.617 | 0.1004–0.1016 | 0.568–0.572 |
| M1d | M1 focal γ=1 / +soft-F1 loss | 0.614 / 0.616 | 0.137 / 0.109 | 0.568 / 0.573 |
| M2 | M1 + card-test & round extras | 0.623 | 0.0999 | 0.577 |
| G0 | GAM (splines on all columns), v2 features | 0.626 | 0.0988 | 0.577 |
| G1 | GAM, splines on continuous only | 0.626 | 0.0989 | 0.578 |
| G1a | G1 − interaction block | 0.598 | 0.1029 | 0.559 |
| G1b | G1 with hour one-hot / knots 5 / knots 12 / l2 1e-3 / 1e-5 | 0.619–0.625 | 0.0989–0.0997 | 0.571–0.579 |
| G2 | G1 + tiny × ratio_neg (card testing) | 0.629 | 0.0986 | 0.581 |
| G3 | G2 + lr<−2.5, night×round, round≥1M | 0.630 | 0.0984 | 0.580 |
| G3x | 21 other single interaction candidates on G1 (new_device × ratio/linear/quadratic, qr × distance, cat × {night, round, new_device, pin}, …) | 0.625–0.627 | 0.0987–0.0990 | 0.576–0.579 |
| G4 | G3 + **unpenalised category × channel bias** (bug fix) | 0.632 | 0.0980 | 0.580 |
| G5 | G4, spline l2 1e-3 / 3e-3 / 1e-2 | 0.632 | 0.0979 | 0.582 / 0.584 / 0.583 |
| G5a | G5 + tiny × {ratio_pos, category, channel, new_device, pin, young, night, tx, log_avg} | 0.631–0.632 | 0.0980–0.0981 | 0.581–0.582 |
| G5b | G1 + 6 card-test hinges (tiny / non-tiny × ratio), instead of G2's single term | 0.628 | 0.0985 | 0.578 |
| G5c | G5 knots 5 / 12, linear-term l2, no cyclic hour | 0.632 | 0.0979 | 0.580–0.583 |
| G6 | **G5 minus 1k / 10k round flags, old card flag (= final features)** | 0.632 | 0.0978 | 0.581 |
| X1 | log-sum-exp mixture, K=2 / K=3 additive scenarios | 0.628 / 0.625 | 0.0986 / 0.0990 | 0.579 / 0.574 |
| X2 | GAM + residual MLP on GAM offset (3 schedules) | 0.631–0.632 | 0.0979–0.0981 | 0.580–0.581 |
| X3 | logit blend GAM : MLP (M2) 0.8 / 0.6 / 0.5 | 0.632 / 0.631 / 0.630 | 0.0979 / 0.0981 / 0.0982 | 0.583 / 0.580 / 0.580 |
| **Final** | G6, l2 3e-3, 3 × 5-fold, fold-average + full refit | **0.633** | **0.0978** | **0.582** |

Second-split checks (CV seed 1 or 2): G2 0.0985 vs G1 0.0989; G4 0.0980; G6 0.0978 vs G5 0.0979. The direction holds.

## score.py calls (1 of 8 allowed)
| # | Files | public | private | Purpose |
|---|---|---|---|---|
| 1 | final v2 (this folder) | 0.5747 | 0.5825 | final report only; not used for any choice |

## What limits further gains
* **Label noise or Bayes error, not the model.** The calibrated GAM's self-implied F1 equals its OOF F1. Its residuals show no structure in the interaction screen, the per-feature calibration checks (χ²/df < 0.7 everywhere) or the residual MLP. The remaining error is the irreducible randomness of a logistic generator at a 4.7 % base rate.
* **Test size.** About 470 positives per split gives F1 an SD of ±0.016. Real model improvements of +0.01 (v1 → v2) can easily show up as −0.006, as they did here on private. Distinguishing models from here on needs the OOF estimate, which has a far smaller SD (about 0.007 F1, or 0.0001 LL paired).
* Possible small remaining gains: the exact functional forms of the generator's terms (for example whether the new-device effect is linear in the ratio or a step). Each candidate I tried changed LL by ≤ 0.0001.

## Compute & fair play
Local: 2 threads, CPU only. aorus-ts: 3 single-thread CPU workers under `~/olympiad_offload/1B_v2/`, with no GPU used; the folder was deleted afterwards. Nothing under `_teacher/` was opened, except running `score.py` as a black box once. No v1 files were modified.
