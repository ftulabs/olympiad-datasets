# 4A · Community rules: comment moderation

**Final scores (score.py, one scoring call):** public **0.9212**, private **0.9266**. Tier: **Gold** (Gold needs ≥ 0.91; the baseline scores ≈ 0.683 private).

Files: `solution.py` (end to end, `--validate` prints the validation table), `solution.ipynb` (the same code in Data / Model / Training & Inference / Predict sections, already executed), `public_submission.csv`, `private_submission.csv`.
Runtime is about 1.5 min for a final-only run and about 4 to 6 min with `--validate`. It runs on CPU with 1 thread and peaks at about 550 MB of RAM.

## 1. Data perspective (the main source of the gain)

* **Splits.** Train has 8 rules × 1000 rows, and each rule's violation rate is between 44% and 58%. Public has 10 rules × 300 rows, two of them unseen (exams, recruitment). Private has 12 rules × 250 rows, four of them unseen (exams, recruitment, accounts, phishing links). The 9 groups are shared across splits.
* **The example columns are a labelled training set for each rule.** For a given rule, the 4 example columns contain the same small pool in every row. Seen rules have 60 positive and 60 negative examples, and the pool is *identical* in train, public and private. Unseen rules have 40 positive and 40 negative. No example ever appears as a `body`, and no example is reused across rules. Pooling the examples over all rows therefore gives a clean few-shot training set for every rule, including the unseen ones. This one observation turns the task from "generalise to unknown rules" into "train a small classifier per rule".
* **Hard negatives.** The negative examples are on-topic but allowed. Examples: a spoiler that starts with `[SPOIL]`, a post that gives *your own* phone number, a warning about phishing links, a post selling your own used item, a question about how to get a hacked account back. A rule's bodies also include violations of *other* rules, and these are labelled 0. A rule-agnostic "is this a bad comment" model trained with leave-one-rule-out reaches only **0.447** AUC. That is anti-informative, and it is the reason a cross-rule "other topic" feature helps.
* **Text noise.** The text is template-generated with teencode (`mik`, `bn`, `khum`), missing diacritics, upper case, repeated letters (`chỉỉỉỉ`), emoji, and disguised contact details (`z@lo`, `za.lo`, `0637.676.809`). Normalisation steps:
  * NFC encoding and lower case.
  * Diacritics stripped (`đ` becomes `d`).
  * Runs of 3 or more repeated characters collapsed.
  * Every digit mapped to `0`.
  * Emoji and punctuation removed.
  * Cue tokens added: `[SPOIL]` at the start, the word "spoil", a phone-like run of 9 or more digits, a short-link or odd TLD, a price, and a digit-count bucket.
* **Label noise.** Duplicate bodies under the same rule disagree in 12.8% of pairs, so about 6 to 7% of labels look flipped. Several "false positives" are obvious violations labelled 0. The AUC ceiling is therefore close to 0.94, and the validation numbers below are near it.
* **Ideas that did not work:** predicting which rule a body is posted under (out-of-fold) gives only **0.589**. Self-training on a rule's unlabelled bodies went from 0.9248 to **0.9219**. Both were dropped.

## 2. Model perspective

1. **Few-shot per-rule scorers.** TF-IDF features are char_wb 2-5 grams plus word 1-2 grams on normalised text. They are fitted unsupervised on every text, about 29k features. For each of the 12 rules, two scorers are applied to all 14k bodies, which gives a 14000 × 12 score matrix:
   * a logistic regression (C = 10) trained on that rule's example pool;
   * a kNN margin: mean cosine to the top-3 positives minus mean cosine to the top-3 negatives.
2. **Rule-conditioned PyTorch stacker** (trained from scratch, shared across rules). Its features are:
   * the own-rule LR and kNN scores, z-scored within the rule so that 40+40 and 60+60 pools are on the same scale;
   * the maximum score of the body under a *different-topic* rule. Rules whose score columns correlate above 0.5 are treated as the same topic, for example the two spoiler rules, or ads vs. used goods vs. fake goods;
   * own minus other.

   The network is an MLP (16 tanh units, dropout, plus a linear skip), trained with AdamW, full batch, 300 epochs, averaged over 5 seeds. Because its inputs do not depend on which rule it is, it transfers to unseen rules.
3. **Supervised per-rule model** (only for the 8 rules seen in train): a logistic regression on that rule's 1000 labelled train rows plus its example pool.

## 3. Training & inference perspective

* **Validation that mimics the test.** Leave-one-rule-out (LORO) for the stacker reproduces the unseen-rule setting: the held-out rule contributes only its examples. Seen rules use stratified 5-fold within each rule. A random split of train would only measure seen rules.
* **Per-rule rank blend.** Only within-rule order matters, so scores are converted to within-rule ranks. Seen rules get 0.75 × rank(supervised) + 0.25 × rank(stacker), with the weight chosen on out-of-fold data: the curve is flat between 0.7 and 0.8. Unseen rules use rank(stacker).
* **Regularisation.** Logistic regressions use L2 with C = 10 (C between 3 and 30 is flat). The stacker uses weight decay, dropout and 5 seeds.
* **Final fit** uses all 8000 train rows, the example pools from all three splits, and TF-IDF fitted on all text, which is unsupervised and transductive. No test labels and no external data are used.

## Ablation log (validation on train labels, column-averaged AUC)

| # | Setting | Overall | Seen-rule protocol | Unseen-rule protocol |
|---|---|---|---|---|
| 0 | Baseline notebook (body-only bag of words, random split) | - | - | private ≈ 0.683 |
| 1 | Rule membership classifier P(rule \| body), OOF | 0.589 | - | 0.589 |
| 2 | Rule-agnostic "badness" model, LORO | 0.447 | - | 0.447 |
| 3 | Few-shot LR on example pool (60+60) | 0.9248 | - | 0.9248 |
| 3a | same, but only 40+40 examples (the real unseen-rule size, 3 subsamples) | 0.9228 | - | 0.9228 |
| 3b | char only / word only / raw+normalised chars / char 1-6 | 0.9229 / 0.9229 / 0.9246 / 0.9244 | - | - |
| 3c | C = 3 / 10 / 30 | 0.9235 / 0.9248 / 0.9253 | 0.9362 / 0.9358 / 0.9338 | - |
| 4 | Few-shot kNN margin (top-3) | 0.9238 | - | 0.9238 |
| 5 | 3 + self-training on unlabelled bodies | 0.9219 | - | 0.9219 |
| 6 | Stacker (LR) own scores + "other-topic" max | 0.9273 | - | 0.9273 |
| 7 | **PyTorch stacker** (z-scored own LR/kNN, other-topic max, diff), LORO | **0.9285** | - | **0.9285** |
| 8 | Supervised per-rule LR (train rows + examples), 5-fold | 0.9358 | 0.9358 | - |
| 9 | **Rank blend 0.75·(8) + 0.25·(7)** | **0.9367** | **0.9367** | - |

Test (score.py): public **0.9212**, private **0.9266**. The test scores sit between the seen-rule and unseen-rule validation numbers, which is what we expect given the share of unseen rules in each split.

## What mattered most

1. Treating the pooled example columns as a labelled few-shot training set for every rule. This alone takes the score from about 0.68 to 0.925, including on unseen rules.
2. Normalisation for diacritics and teencode, plus explicit cue tokens (`[SPOIL]` prefix, phone-like digits, short links, prices) on top of char n-grams.
3. Using train labels per rule where possible (0.925 to 0.936 for seen rules), combined by a per-rule rank blend.
4. A rule-conditioned stacker that adds "this looks more like a *different* rule's violation", which encodes the hard-negative structure (0.925 to 0.9285 LORO).
