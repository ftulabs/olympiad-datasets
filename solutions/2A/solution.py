"""2A Traffic camera detection - end-to-end train + predict.

Faster R-CNN MobileNetV3-Large FPN (COCO-pretrained, torchvision; FPN extended to stride 8/16) trained on 320x320 synthetic
traffic frames upscaled to IMG_SIZE, with test-like augmentation (zoom-out mosaic, rot90,
flips, darkening, blur, noise, haze), small anchors, cosine LR, low score threshold,
rot90/flip TTA (4 views) fused per class.

Usage (final submission = the default command, ~35 min on a GTX 1650):
  python solution.py               # train on 1800 imgs, val on 200 (plain + test-like), predict
  python solution.py --mode full   # (untested) train on all 2000 imgs, then predict

Note: AMP is NOT used - on this GTX 1650 fp16 autocast was much slower than fp32.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision.models.detection import (  # noqa: E402
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models import mobilenet_v3_large  # noqa: E402
from torchvision.models.detection import (  # noqa: E402
    FasterRCNN,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator  # noqa: E402
from torchvision.models.detection.backbone_utils import _mobilenet_extractor  # noqa: E402
from torchvision.ops.misc import FrozenBatchNorm2d  # noqa: E402
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # noqa: E402
from torchvision.ops import batched_nms  # noqa: E402

cv2.setNumThreads(1)
torch.backends.cudnn.benchmark = True
torch.set_num_threads(2)

DATA_DIR = Path("/home/minh/Desktop/olympiad_ai/warmup/2A_traffic_detection/dataset")
OUT_DIR = Path(__file__).resolve().parent
CKPT_DIR = Path("/tmp/2A_ckpt")
CLASSES = ["motorbike", "car", "bus_truck", "bicycle", "pedestrian"]
CLASS_TO_ID = {c: i + 1 for i, c in enumerate(CLASSES)}
BOX_COLS = ["x_min", "y_min", "x_max", "y_max"]
SUB_COLUMNS = ["image_id", "class", "score", *BOX_COLS]
NATIVE = 320
SEED = 42

log = logging.getLogger("2A")


# ----------------------------------------------------------------------------- data
def load_image(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def clip_boxes(boxes: np.ndarray, labels: np.ndarray, orig_area: np.ndarray, w: int, h: int,
               min_vis: float = 0.45) -> tuple[np.ndarray, np.ndarray]:
    """Clip boxes to the canvas and drop boxes that are mostly cut off or tiny."""
    if len(boxes) == 0:
        return boxes.reshape(0, 4), labels
    b = boxes.copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    keep = (bw >= 1.5) & (bh >= 1.5) & (bw * bh >= min_vis * orig_area)
    return b[keep], labels[keep]


def rot90(img: np.ndarray, boxes: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Rotate image counter-clockwise by k*90 degrees (np.rot90 convention) with boxes."""
    for _ in range(k % 4):
        h, w = img.shape[:2]
        img = np.rot90(img)
        # CCW: new_x = y, new_y = w - x
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        boxes = np.stack([y1, w - x2, y2, w - x1], 1) if len(boxes) else boxes
    return np.ascontiguousarray(img), boxes


def photometric(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Test-like appearance shift: darker frames, contrast/colour jitter, blur, noise, haze."""
    x = img.astype(np.float32)
    if rng.random() < 0.8:
        x = x * rng.uniform(0.5, 1.15) + rng.uniform(-15, 10)
    if rng.random() < 0.5:
        m = x.mean()
        x = (x - m) * rng.uniform(0.7, 1.3) + m
    if rng.random() < 0.3:
        gray = x.mean(2, keepdims=True)
        x = gray + (x - gray) * rng.uniform(0.5, 1.4)
    if rng.random() < 0.3:
        x = x * np.array([rng.uniform(0.85, 1.15) for _ in range(3)], np.float32)
    if rng.random() < 0.15:  # haze / fog
        a = rng.uniform(0.15, 0.45)
        x = x * (1 - a) + a * rng.uniform(140, 210)
    x = x.clip(0, 255)
    if rng.random() < 0.2:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.3))
    if rng.random() < 0.15:  # motion blur
        k = rng.choice([3, 5, 7])
        ker = np.zeros((k, k), np.float32)
        if rng.random() < 0.5:
            ker[k // 2, :] = 1.0 / k
        else:
            ker[:, k // 2] = 1.0 / k
        x = cv2.filter2D(x, -1, ker)
    if rng.random() < 0.3:
        x = x + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(3, 12), x.shape)
    return x.clip(0, 255).astype(np.uint8)


class TrafficDataset(Dataset):
    """Train dataset with zoom-out mosaic, zoom-in crop, rot90/flip and photometric aug."""

    def __init__(self, img_dir: Path, ids: list[str], labels: pd.DataFrame | None,
                 train: bool, p_mosaic: float = 0.5, aug: bool = True) -> None:
        self.img_dir, self.ids, self.train = img_dir, list(ids), train
        self.p_mosaic, self.aug = p_mosaic, aug
        self.ann: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if labels is not None:
            for k, g in labels.groupby("image_id"):
                self.ann[k] = (g[BOX_COLS].values.astype(np.float32),
                               g["class"].map(CLASS_TO_ID).values.astype(np.int64))

    def __len__(self) -> int:
        return len(self.ids)

    def _get(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        iid = self.ids[i]
        b, l = self.ann.get(iid, (np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)))
        return load_image(self.img_dir / f"{iid}.jpg"), b.copy(), l.copy()

    def _mosaic(self, i: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """2x2 mosaic of downscaled frames -> objects smaller and denser (zoom-out cameras)."""
        canvas = np.zeros((NATIVE, NATIVE, 3), np.uint8)
        cx, cy = int(rng.uniform(0.3, 0.7) * NATIVE), int(rng.uniform(0.3, 0.7) * NATIVE)
        idxs = [i] + [rng.randrange(len(self.ids)) for _ in range(3)]
        regions = [(0, 0, cx, cy), (cx, 0, NATIVE, cy), (0, cy, cx, NATIVE), (cx, cy, NATIVE, NATIVE)]
        all_b, all_l = [], []
        for idx, (x0, y0, x1, y1) in zip(idxs, regions):
            img, b, l = self._get(idx)
            img, b = rot90(img, b, rng.randrange(4))
            s = rng.uniform(0.45, 0.85)
            sz = int(round(NATIVE * s))
            img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_AREA)
            b = b * (sz / NATIVE)
            rw, rh = x1 - x0, y1 - y0
            ox = rng.randint(0, max(0, sz - rw))
            oy = rng.randint(0, max(0, sz - rh))
            crop = img[oy:oy + rh, ox:ox + rw]
            canvas[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            b = b - np.array([ox, oy, ox, oy], np.float32) + np.array([x0, y0, x0, y0], np.float32)
            b, l = clip_boxes(b, l, area, NATIVE, NATIVE) if len(b) else (b, l)
            # clip to region
            if len(b):
                bb = b.copy()
                bb[:, [0, 2]] = bb[:, [0, 2]].clip(x0, min(x1, x0 + crop.shape[1]))
                bb[:, [1, 3]] = bb[:, [1, 3]].clip(y0, min(y1, y0 + crop.shape[0]))
                a2 = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])
                keep = a2 >= 0.45 * (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                b, l = bb[keep], l[keep]
            all_b.append(b)
            all_l.append(l)
        return canvas, np.concatenate(all_b).reshape(-1, 4), np.concatenate(all_l)

    def _single(self, i: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        img, b, l = self._get(i)
        img, b = rot90(img, b, rng.randrange(4))
        if rng.random() < 0.4:  # zoom-in crop
            s = rng.uniform(1.0, 1.5)
            sz = int(round(NATIVE * s))
            img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_LINEAR)
            b = b * (sz / NATIVE)
            ox, oy = rng.randint(0, sz - NATIVE), rng.randint(0, sz - NATIVE)
            img = img[oy:oy + NATIVE, ox:ox + NATIVE]
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            b, l = clip_boxes(b - np.array([ox, oy, ox, oy], np.float32), l, area, NATIVE, NATIVE)
        return img, b, l

    def __getitem__(self, i: int):
        iid = self.ids[i]
        if not self.train:
            img, b, l = self._get(i)
        else:
            rng = random.Random(np.random.randint(1 << 30))
            if self.aug and rng.random() < self.p_mosaic:
                img, b, l = self._mosaic(i, rng)
            elif self.aug:
                img, b, l = self._single(i, rng)
            else:
                img, b, l = self._get(i)
            if self.aug:
                if rng.random() < 0.5:
                    img = np.ascontiguousarray(img[:, ::-1])
                    if len(b):
                        b = np.stack([NATIVE - b[:, 2], b[:, 1], NATIVE - b[:, 0], b[:, 3]], 1)
                img = photometric(img, rng)
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255)
        target = {"boxes": torch.as_tensor(np.asarray(b, np.float32).reshape(-1, 4)),
                  "labels": torch.as_tensor(np.asarray(l, np.int64))}
        return t, target, iid


def make_shifted_val(ds: TrafficDataset, out_dir: Path) -> pd.DataFrame:
    """Deterministic test-like copy of the hold-out: rot90 + darkening + half zoom-out mosaics.
    Mosaic partners are drawn only from the hold-out itself (no train leakage)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    state = np.random.get_state()
    for i, iid in enumerate(ds.ids):
        rng = random.Random(1000 + i)
        np.random.seed(1000 + i)
        img, b, l = ds._mosaic(i, rng) if i % 2 == 0 else ds._single(i, rng)
        img = photometric(img, rng)
        cv2.imwrite(str(out_dir / f"{iid}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        for bb, ll in zip(b, l):
            rows.append([iid, CLASSES[ll - 1], *bb.tolist()])
    np.random.set_state(state)
    return pd.DataFrame(rows, columns=["image_id", "class", *BOX_COLS])


def collate(batch):
    return tuple(zip(*batch))


# ----------------------------------------------------------------------------- model
def build_model(img_size: int, arch: str = "mbv3") -> torch.nn.Module:
    """COCO-pretrained Faster R-CNN. 'mbv3': MobileNetV3-Large FPN extended with stride-8/16
    pyramid levels (stock model only uses stride-32 features - far too coarse for 7 px objects).
    'r50v2': ResNet50-FPN v2 (better, but ~4x slower on the GTX 1650)."""
    kw = dict(min_size=img_size, max_size=img_size, box_detections_per_img=100,
              box_score_thresh=0.001, box_nms_thresh=0.5,
              rpn_pre_nms_top_n_test=2000, rpn_post_nms_top_n_test=1000,
              rpn_pre_nms_top_n_train=2000, rpn_post_nms_top_n_train=1000,
              box_batch_size_per_image=256)
    if arch == "r50v2":
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, **kw)
        model.rpn.anchor_generator = AnchorGenerator(
            sizes=((16,), (32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    else:
        ref = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)
        sd = ref.state_dict()
        bb = _mobilenet_extractor(mobilenet_v3_large(norm_layer=FrozenBatchNorm2d), True, 6,
                                  returned_layers=[2, 3, 4, 5])
        # 15 anchors / location like the pretrained RPN head, but sized per pyramid level
        base = (8, 11, 16, 22, 32)
        sizes = tuple(tuple(int(b * 2 ** lvl) for b in base) for lvl in range(5))
        ag = AnchorGenerator(sizes=sizes, aspect_ratios=((0.5, 1.0, 2.0),) * 5)
        model = FasterRCNN(bb, num_classes=91, rpn_anchor_generator=ag, **kw)
        new_sd = {}
        for k, v in sd.items():  # pretrained FPN blocks 0,1 (stride-32 maps) -> new indices 2,3
            for blk in ("inner_blocks", "layer_blocks"):
                for old, nw in (("0", "2"), ("1", "3")):
                    pre = f"backbone.fpn.{blk}.{old}."
                    if k.startswith(pre):
                        k = f"backbone.fpn.{blk}.{nw}." + k[len(pre):]
                        break
            new_sd[k] = v
        missing, _ = model.load_state_dict(new_sd, strict=False)
        log.info("mbv3 fresh params: %s", missing)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, len(CLASSES) + 1)
    return model


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def predict(model: torch.nn.Module, img_dir: Path, ids: list[str], device: torch.device,
            tta: bool = True, bs: int = 4, views: list[tuple[int, bool]] | None = None,
            merge_iou: float = 0.55) -> pd.DataFrame:
    model.eval()
    ds = TrafficDataset(img_dir, ids, None, train=False)
    dl = DataLoader(ds, batch_size=bs, num_workers=2, collate_fn=collate)
    rows = []
    for imgs, _, iids in dl:
        imgs = [im.to(device) for im in imgs]
        variants = views or ([(0, False)] + ([(0, True), (1, False), (3, False)] if tta else []))
        outs_all: list[list[dict]] = []
        for k, flip in variants:
            x = [torch.rot90(im, k, dims=(1, 2)) for im in imgs]
            if flip:
                x = [im.flip(-1) for im in x]
            outs = model(x)
            fixed = []
            for o in outs:
                b = o["boxes"].float().clone()
                if flip:
                    b = torch.stack([NATIVE - b[:, 2], b[:, 1], NATIVE - b[:, 0], b[:, 3]], 1)
                # undo CCW rot by k: apply CW k times. CW inverse of new=(y, W-x): x=W-new_y, y=new_x
                for _ in range(k):
                    b = torch.stack([NATIVE - b[:, 3], b[:, 0], NATIVE - b[:, 1], b[:, 2]], 1)
                fixed.append({"boxes": b, "scores": o["scores"].float(), "labels": o["labels"]})
            outs_all.append(fixed)
        for j, iid in enumerate(iids):
            parts = [oa[j] for oa in outs_all]
            boxes, scores, labels = merge(parts, iou=merge_iou)
            for bb, s, lab in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                rows.append([iid, CLASSES[lab - 1], round(s, 5), *[round(v, 2) for v in bb]])
    return pd.DataFrame(rows, columns=SUB_COLUMNS)


def merge(parts: list[dict], iou: float = 0.55, top: int = 100):
    """Weighted box fusion-lite: per-class NMS clusters, boxes averaged by score,
    score = sum of cluster scores / number of TTA views."""
    if len(parts) == 1:
        o = parts[0]
        keep = torch.argsort(o["scores"], descending=True)[:top]
        return o["boxes"][keep].cpu(), o["scores"][keep].cpu(), o["labels"][keep].cpu()
    b = torch.cat([p["boxes"] for p in parts])
    s = torch.cat([p["scores"] for p in parts])
    lab = torch.cat([p["labels"] for p in parts])
    n = len(parts)
    keep = batched_nms(b, s, lab, iou)
    from torchvision.ops import box_iou
    ious = box_iou(b[keep], b)
    same = (lab[keep][:, None] == lab[None, :]) & (ious >= iou)
    w = same.float() * s[None, :]
    fused_b = (w @ b) / w.sum(1, keepdim=True).clamp_min(1e-9)
    # each view contributes at most its best matching box to the cluster score
    view = torch.cat([torch.full((len(p["scores"]),), i, device=b.device) for i, p in enumerate(parts)])
    sc = torch.zeros(len(keep), n, device=b.device)
    sc.scatter_reduce_(1, view[None, :].expand(len(keep), -1).long(), w, reduce="amax")
    fused_s = sc.sum(1) / n
    order = torch.argsort(fused_s, descending=True)[:top]
    return fused_b[order].cpu(), fused_s[order].cpu(), lab[keep][order].cpu()


# ----------------------------------------------------------------------------- metric
def box_iou_np(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    iy = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = ix * iy
    union = ((box[2] - box[0]) * (box[3] - box[1])
             + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter)
    return inter / np.maximum(union, 1e-9)


def mean_average_precision(pred: pd.DataFrame, gt: pd.DataFrame, iou_threshold: float = 0.5,
                           per_class: bool = False):
    """Same VOC all-point mAP@0.5 as the baseline notebook."""
    aps = {}
    for c in CLASSES:
        g = gt[gt["class"] == c]
        if len(g) == 0:
            continue
        p = pred[pred["class"] == c].sort_values("score", ascending=False, kind="mergesort")
        gt_boxes = {k: v[BOX_COLS].values.astype(float) for k, v in g.groupby("image_id")}
        used = {k: np.zeros(len(v), bool) for k, v in gt_boxes.items()}
        tp = np.zeros(len(p), bool)
        for i, (image_id, *box) in enumerate(p[["image_id", *BOX_COLS]].itertuples(index=False)):
            if image_id in gt_boxes:
                ious = np.where(used[image_id], -1.0, box_iou_np(np.array(box, float), gt_boxes[image_id]))
                j = int(np.argmax(ious))
                if ious[j] >= iou_threshold:
                    used[image_id][j] = tp[i] = True
        ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
        recall = np.concatenate([[0.0], ctp / len(g), [1.0]])
        precision = np.concatenate([[0.0], ctp / np.maximum(ctp + cfp, 1), [0.0]])
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        k = np.nonzero(recall[1:] != recall[:-1])[0]
        aps[c] = float(np.sum((recall[k + 1] - recall[k]) * precision[k + 1]))
    m = float(np.mean(list(aps.values())))
    return (m, aps) if per_class else m


# ----------------------------------------------------------------------------- train
def train(args: argparse.Namespace) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    labels = pd.read_csv(DATA_DIR / "train/labels.csv")
    train_dir = DATA_DIR / "train/images"
    all_ids = sorted(p.stem for p in train_dir.glob("*.jpg"))
    perm = np.random.default_rng(SEED).permutation(len(all_ids))
    n_val = 200
    val_ids = [all_ids[i] for i in perm[:n_val]]
    tr_ids = all_ids if args.mode == "full" else [all_ids[i] for i in perm[n_val:]]

    val_plain_gt = labels[labels.image_id.isin(val_ids)]
    shift_dir = Path("/tmp/2A_valshift")
    shift_gt = make_shifted_val(TrafficDataset(train_dir, val_ids, labels, train=True), shift_dir)

    ds = TrafficDataset(train_dir, tr_ids, labels, train=True, p_mosaic=args.p_mosaic, aug=not args.no_aug)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=2, collate_fn=collate,
                    drop_last=True, persistent_workers=True)
    model = build_model(args.img_size, args.arch).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    total = args.epochs * len(dl)
    warm = min(300, len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(it, total) / total)))
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"{args.tag}.pt"
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for it, (imgs, tgts, _) in enumerate(dl):
            imgs = [im.to(device, non_blocking=True) for im in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            loss = sum(model(imgs, tgts).values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            tot += loss.item()
            if it % 100 == 0:
                log.info("ep %d it %d/%d loss %.3f lr %.4f %.0fs", ep, it, len(dl), loss.item(),
                         opt.param_groups[0]["lr"], time.time() - t0)
        msg = f"epoch {ep} loss {tot / len(dl):.4f} time {time.time() - t0:.0f}s"
        if args.mode == "val" and (ep % args.eval_every == 0 or ep == args.epochs):
            m_plain = mean_average_precision(predict(model, train_dir, val_ids, device, tta=False), val_plain_gt)
            m_shift = mean_average_precision(predict(model, shift_dir, val_ids, device, tta=False), shift_gt)
            msg += f" | val plain {m_plain:.4f} shifted {m_shift:.4f}"
            score = 0.5 * (m_plain + m_shift)
            if score > best:
                best = score
                torch.save(model.state_dict(), ckpt)
                msg += " *"
        log.info(msg)
    if args.mode == "full":
        torch.save(model.state_dict(), ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    if args.mode == "val":
        for tta in (False, True):
            mp, ap_p = mean_average_precision(predict(model, train_dir, val_ids, device, tta=tta),
                                              val_plain_gt, per_class=True)
            ms, ap_s = mean_average_precision(predict(model, shift_dir, val_ids, device, tta=tta),
                                              shift_gt, per_class=True)
            log.info("FINAL tta=%s plain %.4f %s | shifted %.4f %s", tta, mp,
                     {k: round(v, 3) for k, v in ap_p.items()}, ms, {k: round(v, 3) for k, v in ap_s.items()})
    if args.predict:
        for split in ("public", "private"):
            ids = pd.read_csv(DATA_DIR / f"{split}_test/{split}_test.csv")["image_id"].tolist()
            sub = predict(model, DATA_DIR / f"{split}_test/images", ids, device, tta=not args.no_tta)
            out = OUT_DIR / f"{split}_submission.csv"
            sub.to_csv(out, index=False)
            assert (sub.image_id.value_counts() <= 100).all()
            log.info("wrote %s (%d boxes, %d images)", out, len(sub), sub.image_id.nunique())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["val", "full"], default="val")
    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--arch", choices=["mbv3", "r50v2"], default="mbv3")
    ap.add_argument("--p-mosaic", type=float, default=0.5)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--tag", default="mb640")
    args = ap.parse_args()
    args.predict = True  # always write the two submissions
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log.info("args %s", vars(args))
    train(args)


if __name__ == "__main__":
    main()
