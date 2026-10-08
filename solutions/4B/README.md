# 4B · Tết demand forecasting

**Final scores (score.py):** public **0.4201** · private **0.4187** → tier **🥇 Gold** (private; public 0.4201 = Silver) (baseline ≈ 0.616 private).

Files: `solution.py` (end-to-end, `python solution.py`; `--backtest 2025` reproduces the validation), `solution.ipynb` (the same code split into Data / Model / Training & Inference / Predict), `public_submission.csv`, `private_submission.csv`.

## Validation design
The notebook's December validation has no Tết in it, so I used **season-aligned backtests that match the real task**: origin 31 Dec of year Y−1, horizon 64 days (1 Jan – 4 Mar Y), training only on windows whose targets end on or before the origin.
- **Fold 2025** (Tết 29 Jan, trained on 2023 + 2024 seasons + generic windows): the main fold.
- **Fold 2024** (Tết 10 Feb, only one earlier season, whose history starts in Nov 2022): a harder secondary check.
WAPE is computed on rows that are not stock-outs, because the test period has none. I also report it on the public-store and private-store splits.

## 1 · Data (about 55% of the effort)
- **Tết dominates, and it follows the lunar calendar.** Revenue aligned on days-to-Tết (k) is very stable across years. Gift items (`ruou_qua`) are 4× baseline at k = −40 and 20–40× at k = −20…−8. Confectionery and drinks run 2–6×. Fresh food peaks at k = −4…−1 (5–6×). Everything drops to 0.2–0.5× in the week after Tết. **On lunar New Year's day every store sells 0.** The solar month cannot capture this because Tết moves between 22 Jan and 17 Feb. Hence a days-to-Tết embedding in [−63, 45].
- **Lunar day 1/15 (and 14, 29, 30)** spikes, mostly in `hoa_le` (flowers, incense): about 3–5× on 1st/15th. Weekday effect is small (Friday +8%).
- **Store types:** student stores (S04, S08) *drop* at Tết and in summer, while residential and market stores rise. **Level shifts:** S03 fell about 25% from June 2025 and has been stable since September. S05 was closed for about 2 weeks in July 2024 (all zeros, masked).
- **New store S10** (opened 15 Sep 2025, private set) and **new item I47** (20 Nov 2025) are still ramping up. **Tết-only items:** I15/I16 (gift boxes), I30 (bánh chưng), and the cold-start **I48** "Tết 2026 new gift box" (450k VND, heavily weighted). In past seasons I15/I16/I30 followed almost identical k-profiles.
- **Promotions:** uplift is about 1 + 4·discount (30% off → 2.2×). For 7 days after a promo, sales dip to about 0.89×.
- **Stock-outs:** zero runs on series with a healthy median (for example 4 zeros at a median of 77). Flagged as zero with a ±14-day median ≥ 4, or a zero run with median ≥ 2. They are masked in the loss and in the level estimates.
- **Features:** a robust baseline level (mean of the last 56 *normal* days: no promo or post-promo, no holiday, outside the Tết zone k ∈ [−45, 20]), with a 14-day window for young series. Also a short/long trend, a store scale, the item mean, discount, a post-promo flag, a young flag and age.

## 2 · Model (about 30%)
- **Direct multi-horizon, season-aligned framing:** one example per (store, item, target day) in a 64-day window after an origin, with features frozen at the origin. That is exactly what the test asks for. Dec-31 origins give Tết windows; month-start origins (Mar–Nov) add generic windows, which teach promo, weekday and lunar effects and give the embeddings more data.
- **Global MLP** (2×192, SiLU) over embeddings of store, item, category, store type, city, weekday, k, lunar day, holiday and horizon week, plus numeric features.
- **Level handling:** output = log(baseline) + MLP(x). The network learns *multipliers*, so scale differences between stores and items do not have to be learned. Tết-only items have no baseline and fall back to item embedding × store scale.
- **Loss matched to the metric:** Poisson NLL weighted by the item's price (= the metric weight), not MSE on log1p (the baseline's choice, which is biased low and ignores revenue).

## 3 · Training & inference (about 15%)
- AdamW with OneCycle LR (max 6e-3), 12 epochs, batch 4096. **5-seed average** (seed averaging alone gave about −0.015 WAPE).
- Final fit uses all history up to 31 Dec 2025 (3 Tết seasons + generic windows).
- **Cold start:** I48 uses I16's embedding, then is rescaled by the ratio of I48 to I16 sales at the same days-to-Tết (Dec 2025 vs earlier seasons). For S10 and I47, a saturating curve L(t) = A(1−e^{−t/τ}) is fitted to weekly levels and the forecast is scaled by the projected growth (capped at +25%).
- Lunar New Year's day is forced to 0. No global shrink factor: a 0.90–1.0 scan on the backtests showed 1.0 is best.

## Ablation log (backtest WAPE, lower is better)
| # | Variant | Fold 2025 | Fold 2024 |
|---|---|---|---|
| 0 | Baseline notebook (solar features, MSE on log) | — (≈0.616 private on the real test) | — |
| 1 | Tết windows only, 6 epochs, 1 seed | 0.502 | 0.611 |
| 2 | Tết windows only, 60 epochs, 1 seed | 0.437 | — |
| 3 | + generic month-start origins, 6 ep, 1 seed | 0.415 | 0.499 |
| 4 | + season-aligned lag feature (last season, same k), 6 ep | 0.427 | 0.503 |
| 5 | (4) with Tết windows weighted ×3 | 0.423 | 0.512 |
| 6 | (4) at 12 epochs | 0.411 | 0.505 |
| 7 | (3) at 12 epochs, no lag feature | 0.400 | 0.489 |
| 8 | (7) with 3 seeds | **0.385** | — |
| 9 | (6) with 3 seeds | 0.389 | — |
| 10 | (8) × shrink 0.93 / 0.96 | 0.387 / 0.385 | — |

On fold 2025 the private-store split was 0.376 and the public-store split 0.393 (row 8). The final model is row 8 with 5 seeds.
Error analysis (row 3): most of the remaining error is in the last pre-Tết week (k = −7…−1), which was under-forecast in 2025, and in the high-price gift items (I15, I16, I07).

## Scores
| | public | private |
|---|---|---|
| Baseline | — | ≈0.616 |
| This solution (score.py, 1 submission scored) | 0.4201 | **0.4187 🥇** |

Final fit: 778k training rows, 5 seeds, about 7 min CPU single-thread, peak RSS about 0.96 GB. Projected ramp-up for S10/I47 came out at 1.00 (their levels have flattened); I48/I16 ratio = 0.635.
The optional remote host (aorus-ts) was not needed and was not used.
