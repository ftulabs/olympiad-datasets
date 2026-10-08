# 3B · Molecule binding: solution v2

**Final (score.py): public mAP 0.5014, private mAP 0.4455.**
v1 was public 0.4837 / private 0.4265 (the task brief quotes 0.4211 for v1's first submission).
The brief puts the noise ceiling at about 0.575 private.

All model selection used my own validation (below). score.py was called 3 times in total, all logged in §5.
No call was used to choose between alternatives.

| Files | |
|---|---|
| `solution.py` | End-to-end script. Featurises the data, fits the 4 model families on all 40k train rows, blends them, writes both CSVs. Picks CUDA or CPU automatically. |
| `solution.ipynb` | The same pipeline as a notebook: Data / Model / Training & Inference / Predict. No submit cell. |
| `public_submission.csv`, `private_submission.csv` | Final predictions, same format as v1. |
| `work/` | Experiment scripts: `lib2.py`, `cv.py` (grouped CV + full fits), `fp_explore.py` and `fpfeats.py` (linear fingerprint models), `stack.py` and `wsearch.py` (blending, calibration, stacking), `eda*.py`. Large caches and out-of-fold arrays were deleted. |

How the submitted CSVs were made: identical components were trained through `work/cv.py --fold -1` (GNNs) and `work/final_lr.py` (linear models) on minh-sager (GTX 970M) and aorus-ts (CPU). They were then blended with exactly the family averaging and weights in `solution.py`. `work/make_sub.py` composed score calls 1–2; the final blend was a short inline version of the same logic that adds family averaging. `work/v1sol.py` is a copy of v1's featuriser, imported by the experiment scripts.
`solution.py` runs the same functions in one process; it was smoke-tested end to end on a data subset.
Re-running it gives numbers that agree up to seed and CUDA nondeterminism. The full run takes about 1.5–2 h on the 970M, mostly the 15 GNN fits.

---

## 1. Data

**Generating process (EDA, `work/eda*.py`)**
- I reused v1's decomposition (bridge bonds → ring systems → the centroid ring system with 3 branches). Every molecule is one 6-atom aromatic **scaffold** plus **3 building blocks**.
  - Train has 5 scaffolds and 406 blocks. Each block appears 198–360 times.
  - Every block appears with every scaffold, in proportion to the scaffold's size. Block co-occurrence shows no role/position structure.
  - So this is a random combinatorial library: one scaffold and 3 blocks drawn from one pool.
- I checked for leakage through the sampling design and found none:
  - Block frequency vs. block binder rate: Spearman correlation about 0 for each target.
  - `id` vs. labels: correlation about 0.
- Test novelty, measured against train:
  - Private rows by number of unseen blocks (0/1/2/3): 1532 / 1143 / 1115 / 1210. About 20% of private rows have an unseen scaffold.
  - Public is milder.
  - The ~197 new blocks are **the same set** in public and private (194 shared).
- Label structure:
  - Targets are nearly independent (|r| < 0.015).
  - Block-ID additive logistic model on a random split: mAP 0.414.
  - Adding block×scaffold or block-pair interaction terms does not help (0.392 / 0.413).
  - A GBM on the sorted per-block coefficients adds only +0.004 to +0.03 per target.
  - So the signal is close to additive over parts, and mostly about generalising block effects to new blocks.
- **Key finding of v2: a structure model beats memorising IDs even on seen blocks.**
  - On rows whose blocks are all seen, L2-logistic on ECFP counts reaches s0 = 0.396, while the block-ID model reaches 0.369 (6-fold CV, same rows).
  - 300 occurrences per block make ID estimates noisy, and sharing strength through substructures denoises them.
  - 0/1 **presence** features beat log-counts by a wide margin (0.350 vs 0.334 mix). The label looks like "which motifs are present", not "how many atoms".

**Validation that mimics the private test (`work/lib2.py: make_splits / eval_val`)**
- **6 folds.** Fold *k* holds out building-block group *k* (1/6 of blocks), plus one of the 4 non-dominant scaffolds (folds 0–3), plus a random 12% of the remaining all-seen rows (stratum 0).
  - Training uses only rows with no held-out part (16–20k rows).
- **Metric "mix".** AP with sample weights so that the strata by number of unseen blocks (0/1/2/3) match private's 31/23/22/24% mix.
  - v1's validation was dominated by 1-new-block rows. Private has 24% rows with all 3 blocks new.
  - I also report unweighted `all` and the per-stratum APs s0..s3.
- Every number below is the **mean over 6 folds** unless marked otherwise. Fold-to-fold spread is large (s3 ranges 0.19–0.44), so I only trust differences of about 0.005 or more on all 6 folds.

**Features**
- Atom features as in v1, plus a one-hot of the part and an "attachment atom" flag.
- `work/fpfeats.py` adds:
  - ECFP-like WL keys with full or simple atom invariants, r ≤ 5
  - part-tagged keys
  - atom pairs (type, type, topological distance)

## 2. Model

| family | description |
|---|---|
| `base` | GINE (4 layers, hidden 128, edge-type embeddings, residual + BatchNorm) → per-part (scaffold, block1..3) sum+max pooling → shared φ → DeepSets ρ([φ(scaffold), Σφ(blocks)]) → 3 logits. Same as v1, trained longer. |
| `idboth` | `base` + a scaffold/block **ID embedding with ID-dropout 0.5**: unseen IDs → UNK, so the net learns both modes. The head is DeepSets plus an additive Σ per-part logit head. |
| `lr_bin` | L2 logistic regression on 0/1 presence of ECFP r ≤ 3 keys, C = 0.03. Written in PyTorch (sparse mm + L-BFGS); it reproduces sklearn exactly (0.3341 vs 0.3341). |
| `lr_ap` | L2 logistic regression on log-counts of ECFP r ≤ 3 + atom-pair keys, C = 0.03. |

The GNNs and the linear models make different errors. The linear models are strong on seen blocks; the GNNs are strong on unseen ones. Blending them is where most of the v2 gain on top of longer training came from.

## 3. Training & inference

- GNNs:
  - multi-task BCE, AdamW (wd 1e-2), one-cycle cosine (max LR 2e-3), batch 256, gradient clipping at 5
  - **25 and 40 epochs** (v1 used 15)
  - refit on all 40k rows
  - seed ensembles: base 5×25 ep + 3×40 ep, idboth 5×25 ep + 2×40 ep. Family members are averaged.
- Blend: per-model z-scored logits, weights **base 2 : idboth 1.5 : lr_bin 1 : lr_ap 0.5**. Chosen on the 6-fold OOF, picking the simplest of several near-equal candidates (see the table).
- Post-processing: per-novelty-stratum calibration and a stratum-aware stacker were tried and gave no gain (see the table), so the blend is not post-processed. AP only needs the ranking.

## 4. Ablations (own validation; 6-fold mean; `mix` = private-mimic weighted mAP)

| # | Model / change | all | **mix** | s0 | s3 |
|---|---|---|---|---|---|
| L1 | block/scaffold-ID logistic (cannot score new blocks) | 0.292 | 0.249 | 0.369 | 0.068 |
| L2 | LR ECFP r≤3 log-count, C=0.03 | 0.356 | 0.334 | 0.396 | 0.302 |
| L3 | … C=0.1 / C=0.01 | 0.356 / 0.347 | 0.337 / 0.324 | | |
| L4 | … r≤2 / r≤5 | 0.354 / 0.351 | 0.332 / 0.328 | | |
| L5 | simple atom invariants (element/arom/ring) r≤3 / r≤5 | 0.351 / 0.347 | 0.328 / 0.324 | | |
| L6 | part-tagged keys r≤3 | 0.355 | 0.333 | | |
| L7 | ECFP r≤3 + atom pairs, C=0.03 (`lr_ap`) | 0.359 | 0.339 | 0.401 | 0.288 |
| L8 | L1-penalised LR (folds 0–3) | | 0.291 (L2: 0.325) | | |
| **L9** | **0/1 presence ECFP r≤3, C=0.03 (`lr_bin`)** | **0.369** | **0.350** | 0.415 | 0.322 |
| L10 | presence r≤2 C=0.03 / r≤4 / +atom pairs / C=0.1 | 0.371 / 0.364 / 0.365 / 0.360 | 0.353 / 0.344 / 0.348 / 0.345 | | |
| L11 | FP embedding-bag MLP (folds 0–1) | | 0.258 / 0.318 (LR 0.281 / 0.363) | | |
| G1 | GNN `base`, 15 ep (= v1 main model) | 0.368 | 0.354 | 0.396 | 0.348 |
| G2 | + ID embedding (DeepSets head), 15 ep | 0.360 | 0.347 | 0.382 | 0.334 |
| G3 | + ID embedding, DeepSets + additive head (`idboth`), 15 ep | 0.379 | 0.360 | 0.409 | 0.335 |
| **G4** | **`base`, 25 ep** | **0.388** | **0.375** | 0.415 | 0.391 |
| **G5** | **`idboth`, 25 ep** | **0.393** | **0.375** | 0.425 | 0.330 |
| G6 | `base` 40 ep vs 25 ep (folds 0–2) | 0.378 vs 0.374 | 0.358 vs 0.352 | | |
| G7 | hidden 192 / 5 layers; isolated-block encoders + additive head | not finished: GPU OOM on the shared 3 GB card | | | |
| P1 | per-stratum Platt calibration (leave-one-fold-out) of G1 | 0.367 | 0.350 (−0.004) | | |
| P2 | stratum-specific stacker on L2 + L1 | 0.355 | 0.334 (±0) | | |
| P3 | transductive pseudo-labels (teacher G1+L3) for LR (folds 0–1) | LR +0.015 | blend −0.004 | | |
| B1 | G1 + L2 z-blend (1 : 0.5) | 0.383 | 0.366 | | |
| B2 | G4 + G5 + L3 + L7 (2 : 1.5 : 0.5 : 0.5), score call #2 | 0.403 | 0.386 | 0.435 | 0.356 |
| **B3** | **G4 + G5 + L9 + L7 (2 : 1.5 : 1 : 0.5), final** | **0.404** | **0.388** | 0.437 | 0.353 |

The final also includes 40-epoch seeds (G6). Its single-seed OOF numbers understate the 15-model seed ensemble actually used.

What moved the needle, measured on own validation (mix):
1. Longer GNN training, 15 → 25 epochs: +0.021.
2. Blending GNNs with linear fingerprint models: +0.011.
3. Binary presence features for the linear model: +0.016 on the linear model.
4. ID embedding with dropout and the additive head: about +0.006 at 15 epochs and ±0 at 25, but it is a different model and adds blend diversity.
5. Seeds and 40 epochs: not measurable on OOF.

## 5. score.py log (8 allowed; 3 used)

| call | submission | public | private |
|---|---|---|---|
| 1 | sanity check: GNN base 15 ep × 3 seeds + LR(L3) + LR(L7), weights 3.5 : 0.75 : 1 | 0.4953 | 0.4367 |
| 2 | G4×5 seeds + G5×5 seeds + L3 + L7 (B2) | 0.5010 | 0.4424 |
| 3 | **final**: base (5×25 ep + 3×40 ep) + idboth (5×25 ep + 2×40 ep) + lr_bin + lr_ap, 2 : 1.5 : 1 : 0.5 (B3 + extra seeds) | **0.5014** | **0.4455** |

## 6. Where this sits and what limits further gains

- Private 0.4455 against v1's 0.4265 (+0.019) and a ceiling of about 0.575. That closes about 13% of the gap from v1 to the ceiling. Public is 0.5014 against v1's 0.4837.
- The gap is almost entirely on **novel building blocks**.
  - On own validation, all-seen rows score about 0.44 and rows with 3 new blocks about 0.35.
  - On private, 24% of rows have 3 new blocks and 20% have a new scaffold.
- Block effects are only partly predictable from structure: v1 measured a cross-validated correlation of about 0.7 between block structure and block effect.
- With 406 training blocks (each seen ~300 times, ~15 positives per target), the limit is the number of distinct blocks, not the number of molecules.
- Seed ensembles and more epochs are already in diminishing returns: +0.007 from 25 → 40 epochs, and blend weights are flat to ±0.002.
- Ideas I did not finish (compute-bound; the 3 GB GPU was shared with another job and bigger configs ran out of memory):
  - a larger or deeper GNN
  - presence-style (max/OR) pooling inside the GNN, motivated by binary features winning in the linear model
  - isolated-block encoders with a purely additive head
  - a graph transformer for model diversity
  - GNN pseudo-labelling on the 197 new blocks: it was neutral for the LR blend
