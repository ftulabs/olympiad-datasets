# 2B v2 · Rice-leaf disease, multi-modal (image + Vietnamese text + tabular)

**Final v2 (score.py): public 0.9412 · private 0.9463.**
v1 was public 0.9201 · private 0.9205, so v2 adds **+0.026 on private**. Own stacking CV: 0.9471 (v1: 0.917).
Gold line 0.84. The "≈0.865 ceiling" in the brief was wrong: v1 had already passed it.

Files: `solution.py` (end-to-end, auto-detects the device, caches each expert in `cache/` so a run can resume),
`solution.ipynb` (the same code as Data / Model / Training & Inference / Predict, with no submit cell),
`public_submission.csv`, `private_submission.csv`, and `work/` (the experiment scripts that produced every number
below: `img_cv.py` is the GPU fold trainer, `runq.py` the retrying GPU queue, `stack.py` the stacker and CV,
`make_pl.py` builds fold-clean pseudo-labels, `experts_tt.py` the text and tab experts, plus small OOF/test
probability arrays). There are no checkpoints.

## Final pipeline
1. **Text expert**: lower-case, strip diacritics, collapse repeated letters, mask province names (`tinhx`), mask
   numbers, and map rare tokens to a frequent token within edit distance 1 (typo normaliser). Then TF-IDF word 1–3 +
   char_wb 2–5 → logistic regression (C=2). OOF 0.641, logloss 0.749.
2. **Tab expert**: 8 numeric columns, soil/variety/season categoricals, a NaN flag, temperature range,
   N per day, rain×storm, humidity−3·tmin → shallow, slow HistGradientBoosting (lr 0.01, 800 iterations, 4 leaves).
   OOF 0.466, logloss 1.232.
3. **Image experts: ImageNet-pretrained ResNet-18** (torchvision weights), avg+max pooled head, run twice:
   at 96 px (seed 1) and upsampled to 112 px (seed 3). Training: 30 epochs, AdamW (backbone lr 5e-4, head ×5,
   wd 1e-2), OneCycle, label smoothing 0.05, batch 32. Augmentation: dihedral (flips + rot90), brightness/colour cast,
   and a separable Gaussian blur with σ∈[0.8,1.8] on 30% of each batch. **8-way dihedral TTA.** The test prediction is the mean
   of the 5 fold models. OOF 0.884 / 0.888.
4. **Fusion**: multinomial LR (C=0.3) on the 4 experts' 5-fold out-of-fold log-probabilities. Text and tab are
   refit on all of train for the test prediction.

The submitted CSVs come from the fold outputs of `work/img_cv.py` on aorus-ts and are stacked with `work/stack.py`.
`solution.py` contains the same code merged into one file. It was smoke-tested end-to-end (`QUICK=1`) on aorus-ts.
A full run takes about 60–80 min on a free GTX 1050 Ti (5 s/epoch for r18 at 96 px), or several hours on CPU.

## Perspective 1: Data (what I found beyond v1)
- **Blur is the main test shift, and it is bimodal.** The Laplacian sharpness has a separate "heavily blurred" mode
  (sharpness < 2.5, roughly a Gaussian blur with σ≈1.2–1.5 on the 96 px leaf). That mode is **14.7% of train but 29.7%
  of test** (both public and private, old and new provinces alike). On blurred photos the image experts lose about
  0.12 F1 (r18: 0.902 on sharp vs 0.767 on blurred), so the test is harder than a stratified CV suggests. I report a
  **blur-reweighted CV (wCV)** that weights blurred train rows to the test's 30%. Blur augmentation was matched to that σ.
- **Unseen provinces** (Cà Mau, Bạc Liêu, 35% of test) are mostly saline soil. The stack predicts more N-deficiency
  there (22% vs 18%), which matches saline → N-def in train. That looks like a genuine covariate effect rather than a
  prior shift to correct, so I did no EM prior re-estimation.
- **Text** is template-generated: greeting + symptom phrase(s) + weather phrase + farmer's guess + filler, with
  teencode and swap/drop typos. The typo rate is identical in train and test (rare-token rate 2.2% / 2.0%). Test has
  fewer accented messages (56% vs 71%), which accent stripping neutralises. Weather phrases ("mưa dầm", "giông lớn")
  echo the tabular weather. The text carries about 0.64 F1 of information, and no feature set I tried moved it
  (table below).
- Remaining confusions in the final stack: healthy ↔ N-deficiency (leaf colour) and blast ↔ brown spot (lesion shape).
  Bacterial blight is almost solved (F1 0.986).

## Perspective 2 + 3: Model / training ablations (own CV, same 5 folds as v1, stratified, seed 0)
Image and text rows are OOF macro-F1 of that expert alone. Stack rows are the LR-stack CV, averaged over 3 stacker-CV
seeds where marked. "blur" is F1 on the blurred subset and wCV is the blur-reweighted CV.

| Experiment | OOF / CV macro-F1 | notes |
|---|---|---|
| Text: v1 config (TF-IDF w1-3 + c2-5, C=2) | 0.6409 (ll 0.7515) | |
| Text: + typo normaliser + number mask (**used**) | 0.6408 (ll 0.7490) | best logloss |
| Text: C = 0.5 / 1 / 4 | 0.647 / 0.651 / 0.633 | higher F1 at C=1 but worse ll (0.763) |
| Text: word 1-4 only / char 1-6 / binary counts | 0.637 / 0.641 / 0.641 | saturated |
| Tab: v1 HGB | 0.4549 (ll 1.2417) | |
| Tab: HGB lr .01, 800 it, 4 leaves | 0.4598 (ll 1.2285) | |
| Tab: + engineered features (**used**) | 0.4655 (ll 1.2323) | |
| Tab: stumps (2 leaves, additive) / LR on binned one-hot | 0.4601 / 0.4263 | |
| Img: v1 tiny CNN-24, 25 ep, 4-TTA | 0.8247 (blur 0.715) | v1 |
| Img: tiny CNN-24, 40 ep, Gaussian-blur aug, 8-TTA | 0.8439 (blur 0.739) | +0.019 |
| Img: tiny CNN-24 + pseudo-labels (fold 0 only, leaky / fold-clean) | 0.834 / 0.831 vs 0.812 | PL helps a from-scratch CNN by ≈+0.02 |
| **Img: ResNet-18 pretrained, 96 px, 30 ep** | **0.8838** (sharp 0.902, blur 0.767) | +0.06 over v1 CNN |
| **Img: ResNet-18 pretrained, 112 px, 30 ep** | **0.8877** (blur 0.785) | |
| Img: ResNet-18 + fold-clean soft pseudo-labels on all 2000 test photos, 20 ep | 0.8771 (blur 0.783) | |
| Stack: text+tab+v1 CNN (v1 reproduction) | 0.9164 (wCV 0.910) | scored v1: 0.920 / 0.921 |
| + blur-gated stacker (Z, Z·blur, blur) | 0.9159 | no gain (also with r18) |
| Stack: text+tab+tiny CNN-24 e40 | 0.9228 | |
| **Stack: text+tab+r18(96)** | **0.9432** (wCV 0.934, blur 0.891) | **scored 0.9405 / 0.9424** |
| + tiny CNN-24 / + v1 CNN | 0.9426 / 0.9417 | the from-scratch CNNs add nothing next to r18 |
| Stacker C = 0.03 / 0.1 / 0.3 / 1 / 3 (3 seeds) | 0.942 / 0.943 / 0.944 / 0.944 / 0.943 | C=0.3 kept |
| Stack: text+tab+r18-PL alone | 0.9350 | PL image model already "contains" text/tab, so stacking double-counts |
| Stack: text+tab+r18 + r18-PL | 0.9435 | no gain → **PL not used** |
| **Stack: text+tab+r18(96)+r18(112) (final)** | **0.9471** (wCV 0.940, sharp 0.954, blur 0.908; 3 seeds) | **scored 0.9412 / 0.9463** |
| same, the two r18s averaged into one expert | 0.9459 | separate experts better |
| final + tiny CNN-24 / + r18-PL | 0.9460 / 0.9467 | no gain |
| final + 3rd r18 (96 px, seed 4; alone 0.8844) as separate expert / seed-averaged with r18(96) | 0.9464 / 0.9470 | no gain: diminishing returns, so I stopped (not scored) |

Notes on what did not work:
- **Pseudo-labelling / co-training.** Fold-clean soft labels for each fold k come from a stack fitted without fold-k
  labels (`make_pl.py`). They give the image model what text and tab know about the 2000 unlabelled test photos. This
  helps the weak from-scratch CNN, but for the pretrained ResNet it lowers the image-only score and does not help the
  stack: the stack already combines the same information, so it is double-counted.
- **Blur gating in the stacker** (log-prob × blur-flag interactions) did not help, even though blurred rows are much
  harder. The LR stack already down-weights an uncertain image expert through its flatter probabilities.
- Prior-shift correction and per-class threshold tuning were not used. v1 found ±0.001 for thresholds, and the test
  class mix the stack predicts matches the train prior apart from the explainable N-def shift.

## score.py calls (2 used of 8)
| # | submission | own CV | public | private |
|---|---|---|---|---|
| 1 | text + tab + r18(96) stack | 0.9432 | 0.9405 | 0.9424 |
| 2 | text + tab + r18(96) + r18(112) stack (**final**) | 0.9471 | 0.9412 | 0.9463 |

Both calls only checked candidates that had already been chosen on CV. Nothing was tuned on the scores.

## Comparison
| | public | private | own CV |
|---|---|---|---|
| baseline notebook | – | ≈0.611 | – |
| v1 (text+tab+tiny CNN stack) | 0.9201 | 0.9205 | 0.917 |
| **v2 (text+tab+2×pretrained ResNet-18 stack)** | **0.9412** | **0.9463** | **0.9471** |

The CV-to-test gap is small (0.947 CV, 0.941 public, 0.946 private), even though test has twice as many blurred
photos. Blur-reweighted CV (0.940) predicted the public score best.

## What limits further gains
- **Blurred photos (30% of test).** The stack is at about 0.954 on sharp photos but only about 0.908 on blurred
  ones. Lesion shape (blast vs brown spot) is largely destroyed by the blur, so those rows fall back to text and tab,
  which carry only about 0.64 and 0.47 F1. Most of the remaining error is probably irreducible. A rough bound if
  sharp rows stay at 0.954: 0.7·0.954 + 0.3·(blur F1). Even a strong blurred-row model would only move the total by
  about 0.01.
- **Text and tab are saturated.** Every feature, regulariser and model variant lands within ±0.005 of the current
  experts. The farmers' self-diagnoses are deliberately unreliable.
- **Compute.** The only GPU is a GTX 1050 Ti with a 1.2 GB cap, shared with another agent's job that ran at 99%
  utilisation the whole time, so r18 took about 8 min per fold. Next steps, in expected value: more ResNet-18 seeds
  and resolutions (each extra diverse expert gave about +0.003 CV), a ResNet-34 or EfficientNet-B0 at 128 px
  (benchmarked: they fit in 1.2 GB at batch 24–32 but cost 2–3× more), and longer schedules (r18 val F1 was still
  rising at epoch 30).

## Compute / fair play
- GPU work ran on aorus-ts (GTX 1050 Ti only, `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`), with
  `torch.cuda.set_per_process_memory_fraction(0.22)`; nvidia-smi showed ≤ 1.10 GB for the process, and the job used
  ≤ 6 CPU threads. When the other agent's job went over its share, the shared card hit OOM, so `work/runq.py` retries a
  failed fold after 60 s. Local work (text, tab, stacking) used 1 thread. Remote scratch `~/olympiad_offload/2B_v2/`
  was removed at the end.
- Nothing under `_teacher/` was read. score.py was used only as a black box, twice. No hidden labels were used for
  training or tuning, and nothing was submitted.
