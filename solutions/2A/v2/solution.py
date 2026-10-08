"""2A Traffic camera detection - v2, end to end (train -> pseudo-label -> self-train -> TTA + WBF).

Model : Faster R-CNN, MobileNetV3-Large FPN (torchvision COCO weights) with the FPN extended to
        stride-8/16 maps and small per-level anchors; input upscaled 320 -> 800.
Data  : test-like augmentation (D4, zoom-out 2x2 tiling s~U(0.6,1), zoom-in up to 1.3, darkening,
        blur, noise, JPEG), pseudo-labelled public+private test frames (self-training, 2 rounds).
Infer : D4 views (id, hflip, rot90, rot270) x scales {800, 960}, weighted boxes fusion, top-100/img.
Val   : v1's 200-image hold-out, scored plain and as a deterministic test-like copy.

    python solution.py                       # full pipeline (~3.5 h on a GTX 1650)
    python solution.py --teacher-csv v1.csv  # skip stage 1, use v1 test predictions as round-1 labels
Device is auto-detected (CUDA if available, else CPU).
"""

from __future__ import annotations

import io
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_large
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _mobilenet_extractor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import box_iou
from torchvision.ops.misc import FrozenBatchNorm2d

CLASSES = ["motorbike", "car", "bus_truck", "bicycle", "pedestrian"]
CLASS_TO_ID = {c: i + 1 for i, c in enumerate(CLASSES)}
BOX_COLS = ["x_min", "y_min", "x_max", "y_max"]
SUB_COLUMNS = ["image_id", "class", "score", *BOX_COLS]
N = 320
SEED = 42
log = logging.getLogger("2Av2")


def find_data_dir() -> Path:
    for c in [os.environ.get("DATA_2A", ""),
              "/home/minh/Desktop/olympiad_ai/warmup/2A_traffic_detection/dataset",
              str(Path.home() / "olympiad_offload/2A_v2/dataset"),
              "../warmup/2A_traffic_detection/dataset"]:
        if c and (Path(c) / "train/labels.csv").exists():
            return Path(c)
    raise FileNotFoundError("dataset not found; set DATA_2A")


def split_ids(data_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """Same 200-image hold-out as v1 (seed-42 permutation) so v1's checkpoint is a calibration point."""
    ids = sorted(p.stem for p in (data_dir / "train/images").glob("*.jpg"))
    perm = np.random.default_rng(SEED).permutation(len(ids))
    val = [ids[i] for i in perm[:200]]
    tr = [ids[i] for i in perm[200:]]
    return ids, tr, val


# ----------------------------------------------------------------------------- augmentation
def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def resize(img: np.ndarray, sz: int) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((sz, sz), Image.BILINEAR if sz > img.shape[0] else Image.BOX))


def d4(img: np.ndarray, b: np.ndarray, k: int, flip: bool) -> tuple[np.ndarray, np.ndarray]:
    """CCW rot90 k times then optional horizontal flip (square images)."""
    s = img.shape[0]
    for _ in range(k % 4):
        img = np.rot90(img)
        if len(b):
            b = np.stack([b[:, 1], s - b[:, 2], b[:, 3], s - b[:, 0]], 1)
    if flip:
        img = img[:, ::-1]
        if len(b):
            b = np.stack([s - b[:, 2], b[:, 1], s - b[:, 0], b[:, 3]], 1)
    return np.ascontiguousarray(img), b


def clip(b: np.ndarray, l: np.ndarray, x0: float, y0: float, x1: float, y1: float,
         min_vis: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    if len(b) == 0:
        return b.reshape(0, 4), l
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    c = b.copy()
    c[:, [0, 2]] = c[:, [0, 2]].clip(x0, x1)
    c[:, [1, 3]] = c[:, [1, 3]].clip(y0, y1)
    w, h = c[:, 2] - c[:, 0], c[:, 3] - c[:, 1]
    keep = (w >= 1.5) & (h >= 1.5) & (w * h >= min_vis * area)
    return c[keep], l[keep]


def photometric(img: np.ndarray, rng: random.Random, strength: float = 1.0) -> np.ndarray:
    """Test-like appearance shift: darker frames, contrast/colour/gamma jitter, haze, blur, noise, jpeg."""
    x = img.astype(np.float32)
    if rng.random() < 0.8 * strength:
        x = x * rng.uniform(0.55, 1.15) + rng.uniform(-15, 10)
    if rng.random() < 0.4 * strength:
        m = x.mean()
        x = (x - m) * rng.uniform(0.7, 1.3) + m
    if rng.random() < 0.3 * strength:
        g = x.mean(2, keepdims=True)
        x = g + (x - g) * rng.uniform(0.5, 1.4)
    if rng.random() < 0.3 * strength:
        x = x * np.array([rng.uniform(0.85, 1.15) for _ in range(3)], np.float32)
    if rng.random() < 0.2 * strength:
        x = 255.0 * (x.clip(0, 255) / 255.0) ** rng.uniform(0.7, 1.5)
    if rng.random() < 0.12 * strength:
        a = rng.uniform(0.15, 0.4)
        x = x * (1 - a) + a * rng.uniform(140, 210)
    x = x.clip(0, 255).astype(np.uint8)
    if rng.random() < 0.25 * strength:
        x = np.asarray(Image.fromarray(x).filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.2))))
    if rng.random() < 0.3 * strength:
        nrng = np.random.default_rng(rng.randrange(1 << 30))
        x = (x.astype(np.float32) + nrng.normal(0, rng.uniform(2, 10), x.shape)).clip(0, 255).astype(np.uint8)
    if rng.random() < 0.3 * strength:
        buf = io.BytesIO()
        Image.fromarray(x).save(buf, "JPEG", quality=rng.randint(55, 90))
        x = np.asarray(Image.open(buf).convert("RGB"))
    return x


class DetDataset(Dataset):
    """items: list of (image_path, boxes[N,4] float32, labels[N] int64).
    train=True: D4, zoom-out mosaic / zoom-in crop, photometric."""

    def __init__(self, items: list, train: bool, scale_lo: float = 0.6, scale_hi: float = 1.3,
                 p_out: float = 0.55, photo: float = 1.0) -> None:
        self.items, self.train = items, train
        self.scale_lo, self.scale_hi, self.p_out, self.photo = scale_lo, scale_hi, p_out, photo

    def __len__(self) -> int:
        return len(self.items)

    def _get(self, i: int, rng: random.Random | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p, b, l = self.items[i]
        img = load(p)
        b, l = b.copy(), l.copy()
        if rng is not None:
            img, b = d4(img, b, rng.randrange(4), rng.random() < 0.5)
        return img, b, l

    def zoom_out(self, i: int, rng: random.Random, s: float, pool: list[int] | None = None):
        """Tile 2x2 randomly-D4'd frames scaled by s, then random-crop the 320 frame."""
        sz = max(8, int(round(N * s)))
        canvas = np.zeros((2 * sz, 2 * sz, 3), np.uint8)
        pool = pool if pool is not None else list(range(len(self.items)))
        bs, ls = [], []
        for t, (gx, gy) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
            idx = i if t == 0 else rng.choice(pool)
            img, b, l = self._get(idx, rng)
            canvas[gy * sz:(gy + 1) * sz, gx * sz:(gx + 1) * sz] = resize(img, sz)
            if len(b):
                bs.append(b * (sz / N) + np.array([gx * sz, gy * sz] * 2, np.float32))
                ls.append(l)
        b = np.concatenate(bs).astype(np.float32) if bs else np.zeros((0, 4), np.float32)
        l = np.concatenate(ls) if ls else np.zeros((0,), np.int64)
        ox = rng.randint(0, max(0, 2 * sz - N))
        oy = rng.randint(0, max(0, 2 * sz - N))
        img = canvas[oy:oy + N, ox:ox + N]
        b = b - np.array([ox, oy, ox, oy], np.float32)
        b, l = clip(b, l, 0, 0, N, N)
        return np.ascontiguousarray(img), b, l

    def zoom_in(self, i: int, rng: random.Random, s: float):
        img, b, l = self._get(i, rng)
        sz = int(round(N * s))
        if sz > N:
            img = resize(img, sz)
            b = b * (sz / N)
            ox, oy = rng.randint(0, sz - N), rng.randint(0, sz - N)
            img = img[oy:oy + N, ox:ox + N]
            b, l = clip(b - np.array([ox, oy, ox, oy], np.float32), l, 0, 0, N, N)
        return np.ascontiguousarray(img), b, l

    def __getitem__(self, i: int):
        if not self.train:
            img, b, l = self._get(i, None)
        else:
            rng = random.Random(np.random.randint(1 << 30))
            if rng.random() < self.p_out:
                img, b, l = self.zoom_out(i, rng, rng.uniform(self.scale_lo, 1.0))
            else:
                img, b, l = self.zoom_in(i, rng, rng.uniform(1.0, self.scale_hi) if rng.random() < 0.6 else 1.0)
            if self.photo > 0:
                img = photometric(img, rng, self.photo)
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255)
        tgt = {"boxes": torch.as_tensor(np.asarray(b, np.float32).reshape(-1, 4)),
               "labels": torch.as_tensor(np.asarray(l, np.int64))}
        return t, tgt, i


def collate(batch):
    return tuple(zip(*batch))


def items_from_labels(img_dir: Path, ids: list[str], labels: pd.DataFrame) -> list:
    g = {k: v for k, v in labels.groupby("image_id")}
    out = []
    for iid in ids:
        v = g.get(iid)
        if v is None:
            out.append((img_dir / f"{iid}.jpg", np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)))
        else:
            out.append((img_dir / f"{iid}.jpg", v[BOX_COLS].values.astype(np.float32),
                        v["class"].map(CLASS_TO_ID).values.astype(np.int64)))
    return out


def make_testlike_val(items: list, ids: list[str], out_dir: Path, seed: int = 7,
                      s_lo: float = 0.75, s_hi: float = 1.0, photo: float = 0.8) -> pd.DataFrame:
    """Deterministic test-like hold-out: every frame randomly D4-rotated (test ~50% vertical roads),
    zoomed out by s~U(s_lo,s_hi) with partner tiles drawn ONLY from the hold-out, test-like photometrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = DetDataset(items, train=True)
    rows = []
    st = np.random.get_state()
    for i, iid in enumerate(ids):
        rng = random.Random(seed * 100003 + i)
        np.random.seed(seed * 1000 + i)
        s = rng.uniform(s_lo, s_hi)
        img, b, l = ds.zoom_out(i, rng, s) if s < 0.999 else ds._get(i, rng)
        img = photometric(img, rng, photo)
        Image.fromarray(img).save(out_dir / f"{iid}.jpg", quality=92)
        for bb, ll in zip(b, l):
            rows.append([iid, CLASSES[ll - 1], *bb.tolist()])
    np.random.set_state(st)
    return pd.DataFrame(rows, columns=["image_id", "class", *BOX_COLS])


# ----------------------------------------------------------------------------- models
COMMON = dict(box_detections_per_img=150, box_score_thresh=0.001, box_nms_thresh=0.5,
              rpn_pre_nms_top_n_test=3000, rpn_post_nms_top_n_test=1500,
              rpn_pre_nms_top_n_train=3000, rpn_post_nms_top_n_train=1500,
              box_batch_size_per_image=512, rpn_batch_size_per_image=256)


def build_model(arch: str, img_size: int) -> torch.nn.Module:
    kw = dict(min_size=img_size, max_size=img_size, **COMMON)
    if arch in ("r50v2", "r50"):
        fn, w = ((fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1) if arch == "r50v2"
                 else (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights.COCO_V1))
        model = fn(weights=w, trainable_backbone_layers=5 if arch == "r50v2" else 4, **kw)
        # two sizes x three ratios per level (P2..P6); RPN cls/bbox layers re-initialised, conv kept
        base = img_size / 320.0
        sizes = tuple((int(round(s * base)), int(round(s * 1.41 * base))) for s in (5, 10, 20, 40, 80))
        model.rpn.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=((0.5, 1.0, 2.0),) * 5)
        old = model.rpn.head
        new = RPNHead(256, 6, conv_depth=len(old.conv))
        new.conv.load_state_dict(old.conv.state_dict())
        model.rpn.head = new
    elif arch == "mbv3":
        ref = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)
        sd = ref.state_dict()
        bb = _mobilenet_extractor(mobilenet_v3_large(norm_layer=FrozenBatchNorm2d), True, 6,
                                  returned_layers=[2, 3, 4, 5])
        base_px = (8, 11, 16, 22, 32)
        sc = img_size / 640.0
        sizes = tuple(tuple(max(4, int(b * 2 ** lvl * sc)) for b in base_px) for lvl in range(5))
        ag = AnchorGenerator(sizes=sizes, aspect_ratios=((0.5, 1.0, 2.0),) * 5)
        model = FasterRCNN(bb, num_classes=91, rpn_anchor_generator=ag, **kw)
        new_sd = {}
        for k, v in sd.items():
            for blk in ("inner_blocks", "layer_blocks"):
                for o, nw in (("0", "2"), ("1", "3")):
                    pre = f"backbone.fpn.{blk}.{o}."
                    if k.startswith(pre):
                        k = f"backbone.fpn.{blk}.{nw}." + k[len(pre):]
                        break
            new_sd[k] = v
        model.load_state_dict(new_sd, strict=False)
    else:
        raise ValueError(arch)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, len(CLASSES) + 1)
    return model


# ----------------------------------------------------------------------------- inference
def undo_d4(b: torch.Tensor, k: int, flip: bool, s: float = N) -> torch.Tensor:
    if flip:
        b = torch.stack([s - b[:, 2], b[:, 1], s - b[:, 0], b[:, 3]], 1)
    for _ in range(k % 4):  # inverse of CCW rot: (x1,y1,x2,y2) <- (s-y2, x1, s-y1, x2)
        b = torch.stack([s - b[:, 3], b[:, 0], s - b[:, 1], b[:, 2]], 1)
    return b


VIEWS = {"1": [(0, False)], "flip": [(0, False), (0, True)],
         "d4": [(0, False), (0, True), (1, False), (3, False)],
         "d8": [(k, f) for k in range(4) for f in (False, True)]}


@torch.no_grad()
def raw_predict(model: torch.nn.Module, paths: list[Path], device: torch.device, views: list,
                sizes: list[int], bs: int = 4) -> list[list[dict]]:
    """Returns per image a list (one per view x size) of dicts boxes/scores/labels (cpu, 320-frame)."""
    model.eval()
    out: list[list[dict]] = [[] for _ in paths]
    orig = model.transform.min_size, model.transform.max_size
    use_amp = device.type == "cuda"
    for sz in sizes:
        model.transform.min_size, model.transform.max_size = (sz,), sz
        for i0 in range(0, len(paths), bs):
            imgs = [torch.from_numpy(load(p)).permute(2, 0, 1).float().div(255).to(device)
                    for p in paths[i0:i0 + bs]]
            for k, flip in views:
                x = [torch.rot90(im, k, dims=(1, 2)) for im in imgs]
                if flip:
                    x = [im.flip(-1) for im in x]
                with torch.autocast(device.type, enabled=use_amp):
                    res = model(x)
                for j, o in enumerate(res):
                    out[i0 + j].append({"boxes": undo_d4(o["boxes"].float().cpu(), k, flip),
                                        "scores": o["scores"].float().cpu(), "labels": o["labels"].cpu()})
    model.transform.min_size, model.transform.max_size = orig
    return out


def _iou_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def wbf(parts: list[dict], iou: float = 0.55, top: int = 100, weights: list[float] | None = None,
        skip: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted boxes fusion (per class, greedy clustering in weighted-score order, at most one box
    per view in a cluster). Fused box = score-weighted mean; fused score = sum_w(score) / sum(weights)."""
    n = len(parts)
    w = np.asarray(weights if weights is not None else [1.0] * n, np.float32)
    b = np.concatenate([p["boxes"].numpy() for p in parts]).astype(np.float32)
    s = np.concatenate([p["scores"].numpy() for p in parts]).astype(np.float32)
    lab = np.concatenate([p["labels"].numpy() for p in parts])
    view = np.concatenate([np.full(len(p["scores"]), i) for i, p in enumerate(parts)])
    ws_all = s * w[view]
    keep0 = s >= skip
    ob, os_, ol = [], [], []
    for c in np.unique(lab[keep0]):
        m = np.nonzero((lab == c) & keep0)[0]
        m = m[np.argsort(-ws_all[m], kind="stable")]
        bc, wc, vc = b[m], ws_all[m], view[m]
        ious = _iou_mat(bc, bc)
        assigned = np.zeros(len(bc), bool)
        for i in range(len(bc)):
            if assigned[i]:
                continue
            cand = np.nonzero((ious[i] >= iou) & ~assigned)[0]
            _, first = np.unique(vc[cand], return_index=True)
            sel = cand[first]
            assigned[sel] = True
            ww = wc[sel]
            ob.append((ww[:, None] * bc[sel]).sum(0) / max(ww.sum(), 1e-9))
            os_.append(ww.sum() / w.sum())
            ol.append(c)
    if not ob:
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)
    ob_a, os_a, ol_a = np.stack(ob), np.asarray(os_), np.asarray(ol)
    order = np.argsort(-os_a, kind="stable")[:top]
    return torch.from_numpy(ob_a[order]), torch.from_numpy(os_a[order]), torch.from_numpy(ol_a[order])


def fuse_to_df(ids: list[str], per_image: list[list[dict]], iou: float = 0.55, top: int = 100,
               weights: list[float] | None = None, pre_top: int = 150) -> pd.DataFrame:
    rows = []
    for iid, parts in zip(ids, per_image):
        parts = [{k: v[:pre_top] for k, v in p.items()} for p in parts]
        if len(parts) == 1:
            p = parts[0]
            o = torch.argsort(p["scores"], descending=True)[:top]
            bb, ss, ll = p["boxes"][o], p["scores"][o], p["labels"][o]
        else:
            bb, ss, ll = wbf(parts, iou=iou, top=top, weights=weights)
        bb = bb.clamp(0, N)
        for x, sc, lb in zip(bb.tolist(), ss.tolist(), ll.tolist()):
            rows.append([iid, CLASSES[int(lb) - 1], round(float(sc), 5), *[round(v, 2) for v in x]])
    return pd.DataFrame(rows, columns=SUB_COLUMNS)


# ----------------------------------------------------------------------------- metric
def _iou1(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    iy = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = ix * iy
    union = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def mean_ap(pred: pd.DataFrame, gt: pd.DataFrame, thr: float = 0.5, per_class: bool = False):
    """VOC all-point mAP@0.5, identical rule to the baseline notebook."""
    aps = {}
    for c in CLASSES:
        g = gt[gt["class"] == c]
        if len(g) == 0:
            continue
        p = pred[pred["class"] == c].sort_values("score", ascending=False, kind="mergesort")
        gb = {k: v[BOX_COLS].values.astype(float) for k, v in g.groupby("image_id")}
        used = {k: np.zeros(len(v), bool) for k, v in gb.items()}
        tp = np.zeros(len(p), bool)
        for i, (iid, *box) in enumerate(p[["image_id", *BOX_COLS]].itertuples(index=False)):
            if iid in gb:
                ious = np.where(used[iid], -1.0, _iou1(np.array(box, float), gb[iid]))
                j = int(np.argmax(ious))
                if ious[j] >= thr:
                    used[iid][j] = tp[i] = True
        ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
        rec = np.concatenate([[0.0], ctp / len(g), [1.0]])
        prec = np.concatenate([[0.0], ctp / np.maximum(ctp + cfp, 1), [0.0]])
        prec = np.maximum.accumulate(prec[::-1])[::-1]
        k = np.nonzero(rec[1:] != rec[:-1])[0]
        aps[c] = float(np.sum((rec[k + 1] - rec[k]) * prec[k + 1]))
    m = float(np.mean(list(aps.values())))
    return (m, aps) if per_class else m


# ============================================================================= pipeline
def train_model(arch: str, img: int, epochs: int, lr: float, bs: int, items: list, dev: torch.device,
                seed: int = 0, workers: int = 2, scale_lo: float = 0.6, scale_hi: float = 1.3,
                p_out: float = 0.55) -> torch.nn.Module:
    """SGD + warm-up + cosine, fp32 (AMP is slower on GTX 1650 / no tensor cores), grad-clip 10."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ds = DetDataset(items, train=True, scale_lo=scale_lo, scale_hi=scale_hi, p_out=p_out)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers, collate_fn=collate,
                    drop_last=True, persistent_workers=workers > 0)
    model = build_model(arch, img).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    total, warm = epochs * len(dl), min(500, len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(it, total) / total))))
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for imgs, tgts, _ in dl:
            imgs = [im.to(dev, non_blocking=True) for im in imgs]
            tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
            loss = sum(model(imgs, tgts).values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            tot += float(loss.item())
        log.info("%s@%d epoch %d/%d loss %.4f", arch, img, ep, epochs, tot / len(dl))
    return model


def test_ids(D: Path) -> dict[str, list[str]]:
    return {s: pd.read_csv(D / f"{s}_test/{s}_test.csv")["image_id"].tolist() for s in ("public", "private")}


def predict_raw(model, D: Path, val_ids: list[str], tl_dir: Path, dev, views, sizes) -> dict:
    res = {"val_plain": raw_predict(model, [D / f"train/images/{i}.jpg" for i in val_ids], dev, views, sizes),
           "val_testlike": raw_predict(model, [tl_dir / f"{i}.jpg" for i in val_ids], dev, views, sizes)}
    for s, ids in test_ids(D).items():
        res[f"test_{s}"] = raw_predict(model, [D / f"{s}_test/images/{i}.jpg" for i in ids], dev, views, sizes)
    return res


def fuse_runs(raws: list[dict], key: str, ids: list[str], weights: list[float], iou: float) -> pd.DataFrame:
    per_img = [[p for r in raws for p in r[key][i]] for i in range(len(ids))]
    w = [wt for r, wt in zip(raws, weights) for _ in r[key][0]]
    return fuse_to_df(ids, per_img, iou=iou, weights=w)


FINAL = {
    "pl_thr": 0.5,          # pseudo-label score threshold (fused D4-TTA score)
    "iou": 0.55,            # WBF clustering IoU
    "sizes": [800, 960],    # multi-scale TTA (x D4 views: id, hflip, rot90, rot270)
    "ensemble": ["student_r2"],  # chosen on test-like val: r2 alone 0.9337 > r1+r2 0.9318-0.9322
    "weights": [1.0],
}


def main() -> None:
    import argparse
    import pickle

    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-csv", default="",
                    help="skip stage 1 and use these test predictions as round-1 pseudo-labels "
                         "(the v2 run used v1's public+private submissions, i.e. exactly stage 1's output)")
    ap.add_argument("--work", default="/tmp/2Av2_work")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--quick", action="store_true", help="1 epoch per stage, smoke test")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    D = find_data_dir()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(D / "train/labels.csv")
    _, tr_ids, val_ids = split_ids(D)
    tl_dir = work / "testlike_val"
    tl_gt = make_testlike_val(items_from_labels(D / "train/images", val_ids, labels), val_ids, tl_dir)
    plain_gt = labels[labels.image_id.isin(val_ids)]
    tr_items = items_from_labels(D / "train/images", tr_ids, labels)
    ep = (lambda e: 1) if a.quick else (lambda e: e)
    tids = test_ids(D)

    def pseudo_items(pl: pd.DataFrame) -> list:
        pl = pl[pl.score >= FINAL["pl_thr"]]
        out = []
        for s, ids in tids.items():
            out += items_from_labels(D / f"{s}_test/images", ids, pl[pl.image_id.isin(ids)])
        log.info("pseudo-labels: %d boxes on %d test images", len(pl), len(out))
        return out

    def report(raw: dict, name: str) -> None:
        for key, gt in (("val_testlike", tl_gt), ("val_plain", plain_gt)):
            m_, pc = mean_ap(fuse_runs([raw], key, val_ids, [1.0], FINAL["iou"]), gt, per_class=True)
            log.info("%s %s mAP %.4f %s", name, key, m_, {k: round(v, 3) for k, v in pc.items()})

    def test_pred(raw: dict) -> pd.DataFrame:
        return pd.concat([fuse_runs([raw], f"test_{s}", ids, [1.0], FINAL["iou"]) for s, ids in tids.items()])

    # ---- stage 1: teacher = v1 recipe (mbv3-FPN @640, 7 epochs, train split) -> round-1 pseudo-labels
    if a.teacher_csv:
        pl1 = pd.read_csv(a.teacher_csv)
    else:
        teacher = train_model("mbv3", 640, ep(7), 0.01, 4, tr_items, dev)
        r_t = predict_raw(teacher, D, val_ids, tl_dir, dev, VIEWS["d4"], [640])
        report(r_t, "teacher")
        pl1 = test_pred(r_t)
        del teacher
        torch.cuda.empty_cache()

    # ---- stage 2: student @800 on train split + pseudo-labelled public/private test frames, 10 epochs
    student = train_model("mbv3", 800, ep(10), 0.01, 4, tr_items + pseudo_items(pl1), dev)
    r_s = predict_raw(student, D, val_ids, tl_dir, dev, VIEWS["d4"], FINAL["sizes"])
    report(r_s, "student")
    torch.save(student.state_dict(), work / "student.pt")

    # ---- stage 3: self-training round 2: fine-tune the student 3 epochs (lr 0.003) on its own pseudo-labels
    random.seed(2)
    np.random.seed(2)
    torch.manual_seed(2)
    pl2 = pseudo_items(test_pred(r_s))
    ds = DetDataset(tr_items + pl2, train=True)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate, drop_last=True,
                    persistent_workers=True)
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.003, momentum=0.9, weight_decay=1e-4)
    total, warm = ep(3) * len(dl), min(500, len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(it, total) / total))))
    for e in range(ep(3)):
        student.train()
        for imgs, tgts, _ in dl:
            imgs = [im.to(dev, non_blocking=True) for im in imgs]
            tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
            loss = sum(student(imgs, tgts).values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
        log.info("round-2 epoch %d done", e + 1)
    r_2 = predict_raw(student, D, val_ids, tl_dir, dev, VIEWS["d4"], FINAL["sizes"])
    report(r_2, "student-r2")
    with open(work / "raws.pkl", "wb") as f:
        pickle.dump({"student": r_s, "student_r2": r_2}, f, protocol=4)

    # ---- final: WBF over the chosen models x D4 views x scales
    raws = [{"student": r_s, "student_r2": r_2}[k] for k in FINAL["ensemble"]]
    weights = FINAL["weights"]
    for key, gt in (("val_testlike", tl_gt), ("val_plain", plain_gt)):
        m_, pc = mean_ap(fuse_runs(raws, key, val_ids, weights, FINAL["iou"]), gt, per_class=True)
        log.info("FINAL %s mAP %.4f %s", key, m_, {k: round(v, 3) for k, v in pc.items()})
    out = Path(a.out)
    for s, ids in tids.items():
        sub = fuse_runs(raws, f"test_{s}", ids, weights, FINAL["iou"])
        assert (sub.image_id.value_counts() <= 100).all()
        sub.to_csv(out / f"{s}_submission.csv", index=False)
        log.info("wrote %s (%d boxes)", out / f"{s}_submission.csv", len(sub))


if __name__ == "__main__":
    main()
