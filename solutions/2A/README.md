# 2A · Traffic camera detection: solution

**Final score (score.py): public mAP@0.5 = 0.8674, private mAP@0.5 = 0.8716. That is Gold (needs ≥ 0.60). The baseline scores about 0.227.**

Files:
- `solution.py` trains and predicts end to end. Run `python solution.py`; it takes about 35 min on the GTX 1650.
- `solution.ipynb` is a readable version split into Data / Model / Training & Inference / Predict.
- `public_submission.csv` and `private_submission.csv` are the final submissions. Each has 100 boxes per image at score ≥ 0.001.

The final submission came from **one training run** (`--mode val`, tag `mb640`): 1,800 training images, 200 held out for validation, 7 epochs. I picked it on my own validation and scored it with score.py **once**. The checkpoint is at `/tmp/2A_ckpt/mb640.pt`, outside the solution folder, and may be cleaned up.

## 1. Data (most of the effort)

### EDA
- Every frame is 320×320.
- There are 65,819 boxes over 2,000 images: about 33 per image on average, at most 76.
- The classes are very unbalanced: motorbike 74%, car 11%, pedestrian 9%, bicycle 3.6%, bus_truck 2.7%.
- **The objects are tiny.**
  - Pedestrians have a median size of 7×6 px, and 15% of all boxes have a side shorter than 8 px.
  - Motorbikes are about 11×21 px, cars about 29×50 px, and buses/trucks up to about 180 px.

### Domain shift between train and test
I compared image statistics and looked at montages of the images.
- **Test frames are darker.** Mean brightness is 84 for public and private test, against 93 for train. The 5th percentile is 58 against 66.
- **Test roads are more often rotated** (horizontal or tilted) than in train, where most roads run vertically.
- **Many test cameras are zoomed out**, so objects are smaller and crowds denser.
- Some test frames have motion blur or haze.

### Augmentation, designed to make train look like test
- Random rot90 (k = 0 to 3) and horizontal flip.
- With p = 0.5, a **2×2 zoom-out mosaic**: each tile is downscaled by 0.45–0.85, rotated at random and cropped into its quadrant. This gives smaller, denser objects.
- Otherwise, a zoom-in crop with p = 0.4 (scale 1.0–1.5).
- Photometric changes: brightness ×0.5–1.15 (biased darker), contrast, saturation, per-channel colour gain, haze, Gaussian blur, motion blur and Gaussian noise.
- In both mosaic and crop, a box is dropped if less than 45% of it stays visible.

### Validation
- 200 images are held out from train (seed 42), and every number is reported on two versions of them:
  - **plain**: the held-out images as they are.
  - **shifted**: a deterministic test-like copy. Half the images are zoom-out mosaics built only from other held-out images, and all of them get rot90 and darkening/blur/noise.
- Checkpoints were chosen on the mean of the two. The shifted score turned out to be the better predictor of the test score: 0.74 on shifted val against about 0.87 on test.
- Data loading: the DataLoader uses 2 workers and reads images from disk (RAM stays under about 1 GB).

## 2. Model
- **Faster R-CNN with a MobileNetV3-Large FPN backbone, using torchvision's COCO-pretrained detection weights.** The baseline pretrained only the backbone on ImageNet; here the detection head is pretrained too.
- **I extended the FPN to stride 8 and stride 16.**
  - The stock torchvision mobilenet-FPN only takes features from stride 32. That is 16 native pixels even at 2× input, which is useless for 7 px pedestrians.
  - I used returned_layers = [2, 3, 4, 5], which gives stride-8, stride-16 and two stride-32 maps, plus a pooled level.
  - The pretrained FPN blocks were remapped to the stride-32 levels. The new blocks are initialised fresh.
- **Input upscaled 2× (320 → 640).**
- **Small anchors, sized per level.** Each level keeps 15 anchors per location, so the pretrained RPN head still loads. Sizes are (8, 11, 16, 22, 32) × 2^level on the 640 input, with aspect ratios 0.5, 1 and 2.
- Detection settings: `box_detections_per_img = 100`, score threshold 0.001, RPN post-NMS 1,000.
- I also tried Faster R-CNN ResNet50-FPN v2 (COCO). It is in the code as `--arch r50v2` but too slow on this GPU, as shown below.

## 3. Training and inference
- SGD with momentum 0.9, lr 0.01, weight decay 1e-4. 300-iteration linear warm-up, then cosine decay. Batch size 4, 7 epochs, gradient clipping at 10.
- **fp32, not AMP.** On this GTX 1650, fp16 autocast was *much slower* than fp32 in a benchmark: ResNet50 at 640 with batch 4 took 1.45 s per iteration in fp32 and had not finished after 100 s in fp16. So I turned AMP off.
- Training time was 3.5–7 min per epoch, depending on load on the shared machine.
- Inference keeps every box above score 0.001, up to 100 per image, because mAP rewards recall.
- **TTA over 4 views:** identity, horizontal flip, rot90 and rot270. The views are fused per class: clusters are formed with NMS at IoU 0.55, boxes in a cluster are averaged weighted by score, and the cluster score is the mean over views of each view's best score.

## Ablation log (validation mAP@0.5; "shifted" is the test-like hold-out)

| # | Setup | plain val | shifted val | Notes |
|---|---|---|---|---|
| 0 | Shipped baseline: mbv3-320, backbone-only pretraining, 320 input, score threshold 0.5 | n/a | n/a | private 0.227 (given) |
| 1 | R50-FPN v2 at 640, AMP | n/a | n/a | 6.8 s/iter, about 50 min per epoch, so abandoned |
| 2 | mbv3 with stride-8/16 FPN, COCO weights, 640, augmentation, 4 epochs (mid-run) | 0.9188 | 0.6996 | |
| 3 | Same run, 7 epochs, no TTA | 0.9324 | 0.7262 | |
| 4 | + flip TTA (2 views) | 0.9336 | 0.7349 | |
| 5 | **+ rot90/flip TTA (4 views), fusion IoU 0.55 (final)** | **0.9369** | **0.7418** | public 0.8674 / private 0.8716 |
| 6 | 4-view TTA, fusion IoU 0.50 | 0.9315 | 0.7378 | |
| 7 | 4-view TTA, fusion IoU 0.65 | 0.9373 | 0.7368 | |

Per-class AP for the final setup on shifted val:

| motorbike | car | bus_truck | bicycle | pedestrian |
|---|---|---|---|---|
| 0.878 | 0.962 | 0.910 | 0.508 | 0.452 |

**Bicycle and pedestrian are the bottleneck.** On plain val they reach 0.88 and 0.83.

### Not run, because time ran out
- An ablation of augmentation against no augmentation. The flag `--no-aug` exists but was never used.
- Training at 800 input.
- The 8-view (D8) TTA and NMS 0.6 variants. That run was stopped when I was asked to wrap up.
- `--mode full`, i.e. retraining on all 2,000 images. It is written but not tested.

## Ideas for v2 (in priority order)
1. **Fix the weak small classes (bicycle and pedestrian).** Options:
   - Larger input: 800, or 960 with batch size 2. At 640, mbv3 takes 0.8 s/iter and 800 takes 1.1 s/iter.
   - Add the stride-4 stage (returned_layers including 1) or anchors down to 6 px.
   - Class-balanced image sampling: oversample images that contain bicycles or pedestrians.
2. **Train longer** (12–15 epochs). The loss was still falling (0.78 at epoch 4, 0.69 at epoch 7) and validation was still rising.
3. **Retrain on all 2,000 images** (`--mode full`) with the same schedule, adding 10% more data.
4. **Arbitrary-angle rotation augmentation.** Some test intersections are tilted, not just rotated by 90°. A box can be rotated by turning its corners into an inscribed ellipse and taking that ellipse's bounding box.
5. **Multi-scale TTA** at 640 and 800, plus D8 views. Weighted box fusion, rather than the NMS-based fusion used now, may add a little.
6. Try ResNet50-FPN v2 if the GPU is less loaded. It was about 3× slower per iteration here even in fp32 (2.6 s/iter at 512 input).
