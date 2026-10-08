# 4A · Community rules: comment moderation — v2

**Final scores (score.py):** public **0.9267**, private **0.9319**. Tier: Gold.

| | public | private |
|---|---|---|
| v1 | 0.9212 | 0.9266 |
| **v2** | **0.9267** | **0.9319** |
| Gain | +0.0055 | +0.0053 |
| Ceiling (label noise, about 0.946) | | gap 0.014 (v1 gap 0.019) |

v2 closes about 28% of the gap that v1 left to the label-noise ceiling.

Files:
* `solution.py`: the whole pipeline. Run `python3 solution.py --validate` to print the validation table.
* `solution.ipynb`: Data / Model / Training & Inference / Predict sections, already executed. Its CSVs are byte-identical to the script's.
* `public_submission.csv`, `private_submission.csv`.

Runtime: about 2.7 min on 2 CPU threads, about 1 GB peak RAM. The device is detected automatically. The stacker is tiny, so the CPU is enough.

## 1. Data perspective: hidden structure of the generator

* **Example pools (from v1).** The 4 example columns are a fixed pool for each rule. Pooled over all rows, each rule has 60+60 labelled comments (seen rules) or 40+40 (unseen rules).
* **Template density is a strong transductive signal.** This is the key new finding. Comments are generated from rule-specific templates (slot fills plus teencode noise). About half the bodies posted under a rule are violations built from that rule's templates. The allowed bodies come from a broad shared distribution: chit-chat, other rules' violations, and hard negatives. So the near neighbours of a violation, over all 14k bodies in train, public and private, are **enriched in the same rule**. An allowed comment's neighbours spread over many rules.
  * The share of same-rule bodies among a body's top-k neighbours reaches AUC 0.58 to 0.62 on its own.
  * Inside the stacker it lifts the unseen-rule protocol from 0.929 to 0.937 or more.
  * It only uses the `rule` and `body` columns of the test files, which are legitimate inputs. No labels are involved.
* **Validation that matches the test.** In the test files an unseen rule has only **250** bodies (accounts, links: private only) or **550** (exams, recruitment: public + private), and a **40+40** pool. Density features depend on rule size, so leave-one-rule-out (LORO) on the natural 1,550-body rules would mislead. The fix is to simulate each train rule as an unseen rule, in "blocks":
  * keep 250 or 550 of its train bodies;
  * **delete its other bodies from the neighbour corpus**;
  * subsample a 40+40 pool and recompute its few-shot scores;
  * draw 3 such blocks per size, plus the natural full block for that rule.
  The stacker is evaluated LORO over blocks. It is also *trained* on these blocks, so training matches the regime it is applied in.
* **Label noise.** I looked for structure in the residual errors by group, id, diacritics, upper case, emoji, repeated letters and length. I found none: the errors look like random flips. Removing likely-noisy training rows (confident-learning style) hurt: 0.9358 to 0.9333/0.9339.

## 2. Model perspective

1. **Few-shot scorers for each of the 12 rules** (from v1), on TF-IDF char_wb 2-5 + word 1-2 of normalised text. There are two scorers: a logistic regression on the pool, and a kNN margin.
   * **New:** a third scorer, LR + "other-topic positives". It adds the positive examples of every rule in a *different* topic as weak negatives (w = 0.3). Rules count as the same topic when their few-shot columns correlate above 0.5. This takes the few-shot LR from 0.9248 to 0.9275 in LORO.
2. **Rule-conditioned features**, built per block. Groups:
   * `v1`: z-scored own LR and kNN, other-topic max, own minus other.
   * `aug`: the augmented LR and its difference.
   * `dens`: same-rule share among the top 5/20/50 cosine neighbours, and the number of distinct rules in the top 20.
   * `top`: top-1 and mean top-5 neighbour similarity.
   * `ndup`: same-rule share among near-duplicates with sim > 0.5 and > 0.7, plus their counts.
   * `grp`: same-rule share among the top 20 neighbours from the same group.
   * `nbr`: own-rule few-shot score averaged over the top 5/20 neighbours.
3. **PyTorch stacker** (trained from scratch): an MLP with 16 tanh units, dropout and a linear skip connection. Training is full-batch AdamW for 300 epochs, averaged over 5 seeds. It does not depend on rule identity, so it transfers to unseen rules.
4. **Supervised per-rule LR** for the 8 seen rules. It is trained on the rule's 1,000 train rows plus its pool. It also uses other-topic pool positives (w 0.3) and other-topic train positives (w 0.1) as weak negatives.

## 3. Training & inference perspective

* **Unseen rules:** the score is the stacker trained on all 56 blocks (8 rules × (2 sizes × 3 draws + full)).
* **Seen rules:** within-rule rank blend of 0.4 × supervised + 0.6 × stacker. On out-of-fold data the weight curve is flat between 0.3 and 0.5.
* **Final fit:** all 8,000 train rows, pools from all three splits, and TF-IDF and the neighbour corpus fitted on all text, which is unsupervised and transductive. No test labels are used anywhere.

## Ablation log (own validation, column-averaged AUC over the 8 train rules)

Protocols:
* **U-full**: LORO with the natural rule size.
* **U-250 / U-550**: simulated unseen-rule blocks (LORO, reduced corpus, 40+40 pool).
* **Seen**: 5-fold within rule.

| # | Setting | U-250 | U-550 | U-full | Seen |
|---|---|---|---|---|---|
| v1 | few-shot LR on pool | - | - | 0.9248 | - |
| v1 | v1 stacker (own LR/kNN, other-topic max) | 0.9227 (60-pool) / 0.9273 (40-pool draws) | 0.9274 | 0.9285 | - |
| v1 | supervised per-rule LR | - | - | - | 0.9358 |
| v1 | v1 blend 0.75 sup + 0.25 stacker | - | - | - | 0.9367 |
| 1 | few-shot + transductive EM (soft, wu 0.1/0.3/1) | - | - | 0.9250/0.9249/0.9248 | - |
| 2 | few-shot + EM hard pseudo-labels | - | - | 0.910-0.921 | - |
| 3 | few-shot + other rules' train positives as negatives (w 0.05) | - | - | 0.9278 | - |
| 4 | **few-shot + other-topic pool positives as negatives (w 0.3)** | - | - | **0.9275** | - |
| 5 | few-shot + *all* other-rule bodies as background negatives | - | - | 0.913-0.924 (worse) | - |
| 6 | stacker v1 + aug | 0.9248 | 0.9279 | 0.9289 | - |
| 7 | stacker v1 + rule-membership P(rule\|body) only | - | - | 0.9353 | - |
| 8 | **stacker v1 + aug + dens (kNN same-rule share)** | - | - | **0.9374** | - |
| 9 | 8 + top | 0.9313 (trained on full blocks) | 0.9361 | 0.9375 | - |
| 10 | 9, stacker trained on simulated blocks | 0.9332 | 0.9366 | 0.9370 | - |
| 11 | 10 + label-aware neighbour shares | 0.9334 | 0.9361 | 0.9372 | - |
| 12 | 10 with 40+40 pools in blocks (base) | 0.9360 | 0.9352 | 0.9361 | - |
| 13 | 12 + ndup + grp | 0.9371 | 0.9376 | 0.9374 | - |
| 14 | **13 + nbr = final stacker** (`--validate` run) | **0.9368** | **0.9380** | **0.9376** | - |
| 15 | 14 + density in char-only / word 1-3 / skeleton space | 0.9365 / 0.9364 / 0.9367 | 0.9372 / 0.9371 / 0.9370 | 0.9373 / 0.9374 / 0.9370 | - |
| 16 | 14 with 6 draws per size instead of 3 | 0.9355 | 0.9397 | 0.9383 | - |
| 17 | 14 + 2nd-stage neighbour smoothing of stacker output | +0.0002 to -0.002 | | | - |
| 18 | 14 + transductive LR refit on stacker pseudo-labels (cross-fitted) | 0.9317-0.9320 | 0.9350-0.9365 | 0.9358-0.9369 | - |
| 19 | stacker gradient boosting (HistGB) instead of MLP | 0.9291 | 0.9346 | 0.9339 | - |
| 20 | sup + other-topic positives as weak negatives | - | - | - | 0.9366 |
| 21 | sup + noisy-label removal (\|OOF−y\| > 0.8/0.9) | - | - | - | 0.9333 / 0.9339 |
| 22 | global stacker with sup score as a feature (row CV) | - | - | - | 0.9398 |
| 23 | **seen blend 0.4 sup + 0.6 final stacker** | - | - | - | **0.9401** |

Expected test score from validation, private (8 seen + 4 unseen rules): (8 × 0.9401 + 2 × 0.9368 + 2 × 0.9380) / 12 ≈ 0.939. The same arithmetic gives ≈ 0.932 for v1, against an actual 0.9266, so the test runs about 0.006 below validation. v2's 0.9319 shows the same offset.

## score.py calls (1 of the 8 allowed)

| # | Submission | public | private |
|---|---|---|---|
| 1 | v2 final (this folder) | 0.9267 | 0.9319 |

All selection was done on own validation. The one call was made after the design was fixed.

## What limits further gains

* **Label noise.** The errors that remain at the top of each rule's ranking are mostly flipped labels, for example obvious shop ads labelled 0 under the ads rule. They show no learnable structure. Seen rules validate at 0.940 against a ceiling of about 0.946.
* **Unseen rules** are now at the seen-rule level in simulation (0.937 to 0.938). Every further transductive idea was flat within ±0.001, which is the validation noise level: other text spaces, label-aware neighbours, smoothing, refitting, more draws, gradient boosting. Ideas tried: EM, refit on pseudo-labels, extra density spaces, graph smoothing.
* **Test vs validation offset (~0.006).** The validation-to-test drop is the same for v1 and v2. It probably comes from the 4 real unseen rules, which overlap with seen ones (recruitment vs "việc nhẹ lương cao" investment scams, phishing links vs ads). With no labels for those rules, it cannot be tuned away without fitting to the hidden labels.
