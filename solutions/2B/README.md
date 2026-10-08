# 2B · Rice-leaf disease, multi-modal — solution

**Final scores (score.py): public 0.9201 · private 0.9205 → Gold (≥0.84).** Baseline ≈0.611 private.
Own 5-fold stacking CV: 0.917 macro-F1. score.py was used twice (one intermediate check, one final run).

Files: `solution.py` (end-to-end), `solution.ipynb` (the same code laid out as Data / Model / Training & Inference / Predict),
`public_submission.csv`, `private_submission.csv`, and `oof/`: small OOF/test probability arrays for every expert, plus remote training logs.
These let a follow-up agent re-stack without retraining.

## Pipeline
1. **Text expert**: lower-case, strip Vietnamese diacritics, collapse repeated letters, mask province names. Then TF-IDF word 1–3-grams + char_wb 2–5-grams → logistic regression (C=2).
2. **Tabular expert**: 8 numeric columns, a NaN flag for `nitrogen_kg_ha`, temperature range, and the categorical columns soil/variety/season that the baseline ignored → HistGradientBoosting. `province` is dropped.
3. **Image expert**: BatchNorm CNN (width 24, stride-2 5×5 stem, 4 stages, avg+max pool), trained from scratch at 96×96 for 25 epochs. AdamW, OneCycle LR, label smoothing 0.05. Augmentation: h/v flips, rot90, brightness/colour-cast jitter, random Gaussian blur on 30% of images. 4-way TTA. On test, the 5 fold models are averaged.
4. **Fusion**: multinomial LR (C=0.3) on the experts' 5-fold out-of-fold log-probabilities (stacking).

## Perspective 1: Data
- **Labels**: healthy 37%, N-deficiency 18%, brown_spot 17%, blast 16%, blight 12%.
- **Images**: synthetic leaves on noisy, colourful backgrounds. Class cues are lesions: blast has spindle lesions, brown spot has round brown spots, blight has a straw-coloured tip or edge, N-deficiency has a yellow-green leaf. Blur varies a lot (Laplacian variance 2–1600). Test images are blurrier on average (≈200 vs 252). This is why I added the blur augmentation. Colour statistics match between train and test.
- **Text**: about half the messages have no diacritics. Others contain teencode ("e", "j", "zậy", "hông"), stretched letters ("quáaaa"), swapped-letter typos and filler. Messages describe symptoms ("vết dài như con thoi" → blast, "giọt nhớt vàng" → blight), but farmers' self-diagnoses are unreliable ("nghi bị đốm nâu" appears in 14% of blast reports). Some messages are vague ("lúa như này có sao không"). The text expert alone tops out at about 0.65.
- **Tabular**: nitrogen is missing in 20% of rows (kept as a flag). Strong signals: storm_last_3d and rainfall → blight; season Đông Xuân → blast; low nitrogen and later sowing → N-deficiency. The categorical columns help.
- **Train/test shift**: Cà Mau and Bạc Liêu make up ≈35% of test but never appear in train. So `province` is dropped and province names in the text are replaced by one token. Test also has slightly more saline soil and is slightly blurrier. No other vocabulary shift was found (I compared test-only n-grams against train).

## Perspective 2: Model, ablation (stratified 5-fold OOF macro-F1 on train)
| Model | OOF macro-F1 |
|---|---|
| Text: word TF-IDF, raw (with accents) + LR | 0.630 |
| Text: word TF-IDF, accent-stripped + LR | 0.650 |
| Text: word + char n-grams, stripped, provinces masked (used) | 0.640–0.652 (logloss 0.75, best calibrated) |
| Tab: HGB on numeric + categorical (used) | 0.455 |
| Hand-crafted colour/blur image features + HGB | 0.455 |
| Image: width-16 CNN, 12 epochs, 3-fold (local CPU) | 0.725 |
| **Image: width-24 CNN, 25 epochs, 5-fold (used)** | **0.825** |
| Image: ResNet-18 ImageNet-pretrained, 10 epochs (only 2/5 folds finished) | fold0 0.847, fold1 0.884 |
| Stack text + tab (LR) | 0.749 |
| Stack text + tab + hand features (HGB) | 0.803 |
| Stack text + tab + CNN-16 | 0.890 (scored: public 0.879, private 0.866) |
| Stack text + CNN-24 (no tab) | 0.909 |
| **Stack text + tab + CNN-24 (final)** | **0.917** (scored: public 0.920, private 0.921) |
| + CNN-16 as an extra expert | 0.918 (not used, within noise) |
| + blur × image-logit interactions / + hand features | 0.918 / 0.918 (no gain) |
| Product of experts instead of LR stacking | about 0.01–0.02 lower |
| Per-class bias tuning for macro-F1 (nested CV) | no gain (±0.001), not used |

## Perspective 3: Training & inference
- OneCycle LR, AdamW, label smoothing, 25 epochs (12 epochs underfits: CNN 0.725 → 0.825).
- Random blur augmentation (motivated by the shift in test blur) and 4-way flip/rot TTA.
- Fold ensembling: the 5 CNN fold models are averaged on test. Text and tab test predictions come from refits on all of train.
- Stacking keeps the predicted class mix close to the train prior (healthy 36%, N-def 20%, …).

## Compute / reproducibility
- Text, tab and stacking ran locally on CPU (2 threads).
- The final width-24 CNN 5-fold run ran on the remote host **aorus-ts** on CPU (4 threads). Its GTX 1050 Ti was unusable: the other agent had filled its memory, so cuDNN would not initialise.
- Training code: `img_cv.py tiny 25 5`. Its logic matches `solution.py`; the only difference is the augmentation RNG (`np.random` vs `default_rng`), so a rerun gives near-identical but not bit-identical predictions.
- `python solution.py` (defaults: `IMG_ARCHS=tiny`, 25 epochs, auto-detect GPU) reproduces the full pipeline. It takes about 15 min on 8 CPU threads, or about 1 h on 2 threads of a loaded machine.
- Smoke-tested locally with `EPOCHS_TINY=1`.

## Unfinished / ideas for v2 (in priority order)
1. **Pretrained ResNet-18 image expert**: run it to completion. `IMG_ARCHS=tiny,r18` is already supported in solution.py. Its folds scored 0.85–0.88 vs 0.80–0.84 for the CNN, so stacking both should add about +0.01–0.02. A GPU makes this cheap; on CPU it costs about 1 min/epoch on 4 threads. Also try the no-maxpool variant (`r18nm` in the experiment code) to keep spatial resolution for small lesions.
2. Train the CNN for more epochs and with 2–3 seeds. The fold score was still rising at epoch 25.
3. Add the image expert's predicted blur or quality, or `hình chụp hơi mờ` ("the photo is a bit blurry") in the text, as gating features in the stacker. A non-linear stacker would be needed for this; the LR interactions did not help.
4. A joint end-to-end fusion net (CNN + text bag-of-n-grams + tab embeddings), trained with the expert OOFs as a baseline to beat.
5. Prior shift: the unseen provinces could have a different class mix. EM prior re-estimation (Saerens) on test predictions is untested.

---
**Note from the orchestrator to the v2 agent:** the "ceiling ≈ 0.865" in your brief was an underestimate (it came from
older, weaker reference experts). v1 already scores 0.9205 private, so the true ceiling is unknown and higher. Keep
pushing (e.g. the unfinished pretrained ResNet-18 branch, longer CNN training, more seeds); judge progress on your own CV.
Also: the GTX 1050 Ti on aorus-ts is now free of the v1 3A jobs, but it is shared with the 3A v2 agent.
