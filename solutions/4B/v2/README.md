# 4B · Tết demand forecasting — v2

**Final (score.py):** public **0.3695** · private **0.3687** (Gold). v1 scored 0.4201 / 0.4187, and the ceiling (oracle median) is about 0.342. v2 closes **65 % of the gap between v1 and the ceiling** on private.

Files: `solution.py` runs end to end with `python solution.py`. It detects the device automatically and accepts `--cfg '{"threads":2}'`. `--fold V25 --model glm` reproduces a backtest. The parallel mode fits one model per process (`--model mlp|glm --tag _sX --cfg '{"seeds":2,"seed0":X}'`) and blends with `--combine`. Also included: `solution.ipynb` (Data / Model / Training & Inference / Predict, no submit cell), `tune.py` (tunes the blend and the median step on OOF backtests), `public_submission.csv` and `private_submission.csv`.

## Validation that mimics the test
The test origin is 31 Dec 2025, which is **Tết − 48 days**. The horizon is 63 days, so k (days to Tết) runs from −47 to +15. v1 validated at 31 Dec origins, which give a different k-range in each year. v2 aligns the folds on Tết:
- **V25**: origin 2024-12-12 (= Tết 2025 − 48), training on 2 earlier seasons. This is the main fold.
- **V24**: origin 2023-12-24 (= Tết 2024 − 48), training on 1 season. It tests robustness with little history.

The metric is revenue-weighted WAPE on rows that are not stock-outs. A training window only uses targets up to the origin. Hidden labels were never used for selection; score.py was called only after the final model was frozen.

**Noise floor.** If sales are simulated as NB(μ, φ) around our own V25 forecast, the floor is 0.333–0.36 for φ between 0.07 and 0.1, which matches the stated ceiling of 0.342. The EDA estimate on raw data, φ ≈ 0.1, also includes weekday and lunar variation, so the true φ is lower. The final V25 score of 0.363 is therefore about 0.03 above the floor.

## 1 · Data
- **The data-generating process is a clean multiplicative factor model.** Tết multipliers by category × week-to-Tết, and by store type × week-to-Tết, are almost identical across 2023, 2024 and 2025: ruou_qua is ×5.2 at k ≈ −14 every year, student stores drop to ×0.45 from k = −21. Store type × weekday is strong (office stores fall to ×0.49 on Sunday, market stores rise to ×1.2 at the weekend). Category × lunar day matters too. The dips follow **k**, not the holiday flag, even though the holiday span differs by year (−4…+4 in 2025, −3…+5 in 2026).
- **Noise:** sales are integers with var ≈ μ + 0.1 μ², i.e. NB-like. The expensive items (I03–I05, I14, I15, I27, I29, I48) sell at low counts, so the **median is the WAPE-optimal point forecast**, not the mean.
- **Stores have their own trends:** S04 and S08 grow strongly, S05 declines, S06 grows, and S03 has a level shift. Per-series levels from 56 days are noisy, so I added a **pooled level**: the 364-day series level × the recent store factor × the recent item factor (each a median over the other dimension). The models receive log(pooled / base) and learn a shrinkage weight.
- **Cold-start structure of the test.** S10 is a new store and carries **20 % of private revenue**. I47 is a new item (about 4 %), and I48 is a new gift box (about 4 %, mapped to I16 × 0.635, as in v1). I simulated a new store in V25 (S06 "opened" 107 days before the origin, as S10 was). That costs about +0.035 WAPE on that store, and borrowing an analog store's embeddings did not help (table below).
- **A bug inherited from v1's framework, found by comparing model families on the final fit.** I47 appeared in training only as rows without a baseline (offset 0), so its item embedding learned an absolute level. At test time I47 has a baseline, so the level was counted twice: forecasts were **2.8×** its December level. Likewise, "age" was measured from the start of the data, so every series looked "young" in the Tết 2023 windows. S10's ramp-up also inflated the MLP's trend features (+40 % against an analog ratio of about 1.1). Fixes:
  - age is measured from the true birth (store `open_date` or the item's first sale);
  - non-seasonal rows without a baseline are dropped from training;
  - I47 borrows I11's interactions;
  - trend features are zeroed for series younger than 150 days.

  On the real test the fix was worth **−0.028 public and −0.017 private** (score.py call 2, made after the fix had already been chosen).

## 2 · Model
- **GLM (new, the strongest single model):** a log-additive factor model in PyTorch. The log forecast is log(level) + a linear term on the numeric features + scalar embeddings for crossed indices: item×k, category×k, store-type×k, store×k, city×k, store-type/store/category/item × weekday, category/item × lunar day, category/store-type × holiday, category/item × discount bucket, category × post-promo, store×item, has-baseline, store/item/young × horizon week, store-type × weekday × has-baseline. Penalties: **strong L2 (1e-3) plus a smoothness penalty on first differences over k (0.1)**. That makes it a hierarchical model: the shared category and store-type profiles carry the signal, and item- and store-specific deviations are shrunk. A new store falls back to its store-type and city profiles automatically.
- **MLP (v1 family, upgraded):** embeddings plus numeric features on a log(level) offset, using the new features (pooled level, short/long trend, horizon) and regularisation (wd 1e-2, dropout 0.1).
- **Loss:** revenue-weighted Poisson NLL for both, consistent with the metric.
- Also tried: a hybrid in which an MLP learns the residual on top of the GLM's log-prediction (no better than the GLM), a wide 3×384 MLP (worse), and extra hierarchical GLM terms (neutral).

## 3 · Training & inference
- **Training windows aligned on Tết:** origins at T − 48 + {−21, −14, −7, 0, +7} for every past season, plus 31 Dec origins and month-start generic origins. This was the biggest single gain on V24 (0.475 → 0.394).
- **Ensemble:** MLP with 6 seeds and GLM with 8 seeds, blended as a weighted geometric mean (0.4 MLP / 0.6 GLM, tuned on V25 + V24 OOF).
- **Post-processing:** I48 analog and ratio; saturating-curve ramp-up for young series (capped at +25 %); zero on lunar New Year's day; then the **Poisson median** of the blended mean, tuned on OOF (q = 0.5, φ = 0 was best; −0.0025).
- **Final refit** on all history up to 31 Dec 2025: 1.03 M training rows. The 6 MLP seeds ran on aorus-ts as 3 processes × 2 seeds × 1 thread (about 6.5 min each). The GLM ran locally (2 threads, 4.3 min, peak RSS 1.4 GB).

## Ablations (own validation, WAPE, lower is better; 3 seeds unless noted)
| # | Variant | V25 | V24 |
|---|---|---|---|
| 0 | v1 config (31 Dec Tết origins + month starts, MLP) | 0.3801 | 0.4746 |
| 1 | + Tết-aligned origins with jitter + extra level features (MLP a) | 0.3758 | 0.3936 |
| 2 | + pooled level feature (MLP b) | 0.3770 | 0.3895 |
| 3 | MLP, denser jitter −28…+14 | 0.3807 | 0.3905 |
| 4 | MLP, Tết windows weighted ×2 | 0.3778 | 0.3885 |
| 5 | MLP, wide 384×3, 16 epochs | (stopped) | 0.3931 |
| 6 | MLP wd 1e-3, dropout 0.15 | 0.3750 | 0.3900 |
| 7 | **MLP wd 1e-2, dropout 0.1 (r2)** | 0.3752 | 0.3892 |
| 8 | MLP, 8 epochs | 0.3763 | 0.3939 |
| 9 | r2 with 6 seeds | 0.3735 | 0.3868 |
| 10 | GLM l2 3e-6, dk 3e-4 | 0.3818 | 0.4001 |
| 11 | GLM l2 3e-5, dk 1e-3 (without → with pooled level, 1 seed) | 0.3811 → 0.3794 | 0.3974 → 0.3969 |
| 12 | GLM l2 1e-4, dk 3e-3 | 0.3772 | 0.3924 |
| 13 | GLM l2 3e-4, dk 1e-2 | 0.3741 | 0.3882 |
| 14 | GLM l2 1e-3, dk 3e-2 | 0.3712 | 0.3854 |
| 15 | GLM l2 3e-3, dk 1e-1 | 0.3698 | 0.3859 |
| 16 | **GLM l2 1e-3, dk 1e-1 (j)**; 8 seeds | 0.3697 | 0.3855 |
| 17 | GLM l2 1e-2, dk 3e-1 (too strong) | 0.3717 | 0.3917 |
| 18 | GLM j + hierarchical terms (category×store-type×k, item×k-week, …) | 0.3692 | 0.3864 |
| 19 | Hybrid GLM → MLP residual (GLM l2 3e-5) | 0.3793 | 0.3964 |
| 20 | Blend 0.4 MLP-r2 (6 seeds) + 0.6 GLM-j | 0.3657 | 0.3797 |
| 21 | **+ Poisson median (final recipe)** | **0.3630** | **0.3774** |
| 22 | Final recipe + young-series/age fix (models refit on folds) | MLP 0.3757, GLM 0.3695 | MLP 0.3900, GLM 0.3876 |
| — | 4-model blend (MLP b, MLP tet-w2, GLM j, GLM k) | 0.3653 | 0.3793 |
| — | Median variants: NB φ 0.1 q 0.55 / φ 0.2 q 0.55 / scale 0.95 | 0.3636 / 0.3633 / 0.3662 | 0.3776 / 0.3774 / 0.3806 |

**New-store simulation** (V25, S06 treated as opened 107 days before the origin; WAPE and bias on S06 only):

| | not new | new, no analog | new, analog store S01 |
|---|---|---|---|
| MLP | 0.388 | 0.416 (+0.047) | 0.421 (+0.065) |
| GLM (l2 3e-5) | 0.382 | 0.429 (−0.041) | 0.433 (+0.047) |

Decision: **no store analog for S10.** The models blend MLP (biased high) and GLM (biased low) on new stores.

On v1's own fold (31 Dec 2024 origin, "D25"), v2's MLP a scores 0.3781, against v1's reported 0.385 for the same 3 seeds.

## score.py calls (2 of 8 used)
| # | Submission | public | private | Purpose |
|---|---|---|---|---|
| 1 | **Final v2** (this folder) | **0.3695** | **0.3687** | final score |
| 2 | The same ensemble before the young-series/I47 fix (scratch files) | 0.3971 | 0.3853 | documents the effect of a fix chosen earlier; no selection |

## Comparison
| | public | private |
|---|---|---|
| Baseline notebook | — | ≈ 0.616 |
| v1 | 0.4201 | 0.4187 |
| **v2** | **0.3695** | **0.3687** |
| Ceiling (oracle median) | — | ≈ 0.342 |

## What limits further gains
- **Irreducible noise:** NB noise with φ ≈ 0.07–0.1 and low counts on the expensive items puts the floor at about 0.34. v2 is about 0.027 above it on private.
- **The Tết peak changes from year to year.** Most of the remaining excess error sits at k = −14…−1, where the daily profile differs between seasons (for example, tuoi_song peaks at ×2.7 in 2023 and 2025 but ×1.9 in 2024). Three seasons cannot pin this down.
- **Cold starts are unvalidatable:** S10 (20 % of private), I47 and I48 have no Tết history. The simulation shows a cost of about +0.03 WAPE on a new store whatever the model.
- **Further levers with small expected gains:** more seeds (the GLM is close to deterministic already), stacking on more folds (only two Tết-aligned folds exist), and NB likelihoods. On V25 the 4-model blend gave only −0.0004 over the 2-model blend.
