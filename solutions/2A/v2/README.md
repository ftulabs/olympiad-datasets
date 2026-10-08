# 2A · Traffic camera detection: v2

**Final v2 score (score.py): public mAP@0.5 = 0.9473, private = 0.9497.**
v1 scored 0.8674 / 0.8716, so v2 adds +0.080 public and +0.078 private.
The ceiling is about 1.0 (synthetic boxes, no label noise), so v2 removes about 61% of the gap v1 left (0.128 → 0.050).

| | public | private | own test-like val | own plain val |
|---|---|---|---|---|
| v1 (mbv3 @640, 7 ep, D4 TTA) | 0.8674 | 0.8716 | 0.848 (single view) | 0.937 (TTA) |
| v2 student, round 1 (call 1) | 0.9418 | 0.9446 | 0.9294 | 0.9758 |
| **v2 final: student after self-training round 2 (call 2)** | **0.9473** | **0.9497** | **0.9337** | **0.9766** |

Files:
- `solution.py`: the full pipeline, end to end. It auto-detects the device.
- `solution.ipynb`: the same code, split into Data / Model / Training & Inference / Predict. It has no submit cell.
- `public_submission.csv`, `private_submission.csv`: the final submissions. They have the same format as v1: 100 boxes per image at score ≥ 0.001, in 320-px frame coordinates.

To reproduce, run `python solution.py`. It takes about 3.5 h on a GTX 1650.
- With `--teacher-csv <v1 public+private concatenated>` it skips stage 1. That is exactly what the scored run did: its round-1 pseudo-labels were v1's submissions, which are stage 1's output.
- Checkpoints go to `/tmp`. None are stored in this folder.

## 1. Data

### EDA beyond v1
I estimated the test object scale from confident v1 detections and compared it with the train labels.
- Per image, the median short side of a motorbike is 9.1 px in test against 10.0 px in train.
- Pedestrians are about 5.7 px against 7 px.
- So **test frames are zoomed out by about 0.85–0.9**. That is milder than v1's mosaic, which used 0.45–0.85.

Other differences:
- **Rotation:** 71% of train frames have vertical roads, against about 46% in test.
- **Brightness:** test frames are darker (mean 84 against 92).
- **Sharpness and noise:** test frames vary more. The Laplacian-variance spread is wider, and noise reaches higher levels.

### Validation that mimics the test
I used the same 200-image hold-out as v1 (seed 42), so v1's checkpoint serves as a calibration point. It is scored in two forms:
- **plain**: the hold-out images as they are.
- **test-like**: a deterministic copy built as follows:
  - Each frame gets a random D4 transform (rotation and flip).
  - It is zoomed out by s ~ U(0.75, 1.0), tiling 2×2 with partner frames drawn **only from the hold-out**.
  - It gets test-like photometric changes (strength 0.8).

Calibration:
- v1's checkpoint scores **0.848** (single view) on this test-like val. Its real test score is about 0.86 (single view) and 0.87 (with TTA). v1's own "shifted" val gave 0.726.
- For the final v2 model, the test-like val gives 0.934 against 0.947–0.950 on test. It stays a slightly pessimistic but well-ranked proxy.
- All selection was done on test-like val, with plain val as a secondary check.

### Augmentation
I wrote it with PIL and numpy only, so it runs on every host.
- D4 transforms: rot90 k = 0–3 and horizontal flip.
- With p = 0.55, a **zoom-out**: four D4-randomised frames, scaled by s ~ U(0.6, 1.0), are tiled 2×2 and then randomly cropped to 320.
- Otherwise, a zoom-in crop with s ~ U(1, 1.3), applied with p = 0.6.
- Photometric changes: brightness (biased darker), contrast, saturation, channel gain, gamma, haze, Gaussian blur, Gaussian noise and JPEG re-encoding at quality 55–90.
- A box is dropped if less than 50% of it stays visible.

### Using all legitimate inputs: self-training on the test images
- The 1,000 unlabelled public and private frames are pseudo-labelled by the model, keeping fused TTA score ≥ 0.5, and added to training. No manual labelling and no hidden labels are involved.
- **Round 1:** labels from v1's predictions, 40,100 boxes.
- **Round 2:** labels from the round-1 student, 42,040 boxes.
  - Pedestrians at score ≥ 0.5 rose from 2,138 to 3,276 between rounds, and bicycles from 748 to 1,039. The student recovers many small objects that v1 missed.

## 2. Model
- **Faster R-CNN with a MobileNetV3-Large FPN backbone, from torchvision COCO weights.**
  - The FPN is re-wired to return stride-8, 16 and 32 maps (v1's idea).
  - Anchors are (8, 11, 16, 22, 32) × 2^level, **scaled with the input size**.
- **Input is 800, not 640.** This is the single most important model change, because pedestrians are about 5–7 px natively.
- Detection settings: 150 detections per image, score threshold 0.001, RPN post-NMS 1,500 for both train and test, and 512 ROI samples per image.
- I tried other families but could not use them:
  - **R50-FPN-v2** with P2 (stride-4) and 2 sizes × 3 ratios of anchors. It runs out of memory at batch 4 at 512 on 4 GB, and is about 3× slower.
  - The remote GPUs were both shared with other jobs. One crash came from CUBLAS allocation running out of memory.
  - A second family was therefore not affordable within the time limit.
- A diversity member, mbv3 @640 trained 4 epochs on the 970M, was evaluated as an ensemble partner. It **hurt** (see the ablation table), so it was dropped.

## 3. Training and inference
- SGD with momentum 0.9 and weight decay 1e-4. Linear warm-up over 500 iterations, then cosine decay to 2%. Gradient clipping at 10.
- Batch size 4. **fp32**: AMP is 2.2× slower on the GTX 1650 (0.74 against 1.97 s/it at 800). This matches v1's finding.
- **Student (round 1):** 1,800 training images plus 1,000 pseudo-labelled test images. 10 epochs at lr 0.01, about 8.3 min per epoch.
- **Round 2:** the student is fine-tuned for 3 more epochs at lr 0.003, on train plus its own round-1 pseudo-labels.
- **TTA:** 4 D4 views (identity, horizontal flip, rot90, rot270) at **2 scales (800 and 960)**, which makes 8 passes.
- **Fusion:** a vectorised **weighted boxes fusion** (WBF).
  - Clusters are built per class, greedily in score order, with at most one box per view per cluster.
  - A fused box is the score-weighted mean of its cluster. Its score is the sum of view scores divided by the number of views, so boxes found in few views are penalised.
  - Then the top 100 per image are kept.
- No refit on the 200 hold-out images: they were needed for selection, and adding 10% more data was worth less than the remaining time.

## Ablation table (own validation, mAP@0.5)

| # | Setup | test-like val | plain val | Notes |
|---|---|---|---|---|
| A | v1 checkpoint (mbv3 @640, 7 ep, no pseudo-labels), single view | 0.848 | 0.932 | calibration; real test ≈0.87 with TTA |
| B | mbv3 @640, 4 ep, round-1 pseudo-labels, new augmentation, single view (970M) | 0.860 | 0.935 | pseudo-labels + augmentation at equal resolution, fewer epochs |
| B2 | B + D4 TTA | 0.8755 | 0.9425 | |
| B3 | B + D4 × {640, 800} | 0.8835 | 0.9443 | |
| C | **mbv3 @800, 10 ep, round-1 pseudo-labels, single view** | 0.9171 | 0.9670 | +0.057 over B: resolution and longer schedule |
| C2 | C + D4 TTA @800 | 0.9245 | 0.9733 | |
| C3 | C + D4 TTA @960 | 0.9283 | 0.9725 | larger TTA scale helps small classes |
| C4 | C + D4 × {800, 960}, WBF IoU 0.55 | 0.9294 | 0.9758 | **score.py call 1: 0.9418 / 0.9446** |
| C5 | C4 with WBF IoU 0.60 / 0.50 | 0.9286 / 0.9296 | 0.9756 / 0.9758 | flat, so I kept 0.55 |
| D | C4 + 0.3 × B3 (2-model WBF) | 0.9238 | 0.9731 | the weaker member hurts |
| D2 | C4 + 0.5 × B3 | 0.9211 | 0.9717 | |
| E | **Round 2: C fine-tuned 3 ep (lr 0.003) on its own pseudo-labels, single view** | 0.9213 | 0.9696 | |
| E2 | E + D4 @800 | 0.9289 | 0.9756 | |
| E3 | **E + D4 × {800, 960} (final)** | **0.9337** | **0.9766** | **score.py call 2: 0.9473 / 0.9497** |
| F | C4 + E3, equal weights / 0.5 × C4 | 0.9318 / 0.9322 | 0.9769 / 0.9771 | the two checkpoints are too correlated |

Final per-class AP on test-like val:

| motorbike | car | bus_truck | bicycle | pedestrian |
|---|---|---|---|---|
| 0.987 | 0.991 | 0.967 | 0.862 | 0.862 |

For comparison, v1 scored bicycle 0.71 and pedestrian 0.68 on the same val.

Caveat on E against C: round 2 adds both better pseudo-labels and 3 extra low-lr epochs. I did not separate the two effects because there was no budget for it.

## score.py calls (2 of 8 allowed)

| call | submission | public | private |
|---|---|---|---|
| 1 | C4: round-1 student, D4 × {800, 960}, WBF | 0.9418 | 0.9446 |
| 2 | **E3: round-2 student, D4 × {800, 960}, WBF (final)** | **0.9473** | **0.9497** |

Both candidates were chosen on test-like val before scoring, and E3 had the higher val score (0.9337 against 0.9294). No hidden-label feedback was used for tuning.

## Compute log
- **Local GTX 1650.** It ran every main run, after v1 had finished.
- **minh-sager GTX 970M.** It was shared with another job and ran only run B.
- **aorus-ts GTX 1050 Ti.** It was used by another job at 3.5 GB / 100%. My R50-v2 attempt failed with CUBLAS allocation running out of memory, so nothing usable ran there.
- Remote scratch `~/olympiad_offload/2A_v2/` was deleted on both hosts after use.

## What limits further gains
1. **The small classes.** Bicycle and pedestrian are at 0.86 on test-like val, while the other three classes are at 0.97–0.99. These objects are 5–8 px. The next steps would be:
   - A stride-4 FPN level for mbv3, or R50-FPN with P2 on a larger GPU.
   - Training at 960 or above.
   - Oversampling images that contain bicycles or pedestrians.
2. **Compute.**
   - Each 800-px epoch takes about 8 min, and TTA inference over 1,400 images × 8 passes takes about 28 min.
   - A different model family (R50-FPN-v2, RetinaNet or FCOS) for a real ensemble would cost hours on a 4 GB card. The same-family partner B was too weak to help.
3. **More self-training rounds.**
   - Round 2 gave +0.004 on val and +0.005 on test.
   - A third round, or re-training from COCO weights on round-2 labels at 960, would probably add another 0.002–0.005.
4. **Refit on all 2,000 images**, putting the hold-out back into training. Expected gain is small, about +0.002.

The remaining gap to 1.0 is about 0.05. Most of it is tiny-object recall and localisation at IoU 0.5 for 5-px boxes, where one pixel of error is 20% of the box side.
