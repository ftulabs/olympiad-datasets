"""Image expert CV trainer for 2B v2 (runs on aorus-ts GPU).

usage: python img_cv.py ARCH EPOCHS SEED [FOLDS=0,1,2,3,4] [extra k=v ...]
writes out/{tag}_f{k}.npz with oof (val idx, probs), test probs, blurred-val probs.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from PIL import Image
from sklearn.model_selection import StratifiedKFold

D = os.environ.get("DATA_DIR", os.path.expanduser("~/olympiad_offload/warmup/2B_rice_multimodal/dataset"))
HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["healthy", "blast", "brown_spot", "bacterial_blight", "nitrogen_deficiency"]
torch.set_num_threads(int(os.environ.get("NT", "3")))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("VRAM_FRAC", "0.28")), 0)
    torch.backends.cudnn.benchmark = True


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_cache() -> dict:
    p = f"{HERE}/cache.npz"
    if os.path.exists(p):
        return dict(np.load(p, allow_pickle=True))
    out = {}
    for s in ["train", "public_test", "private_test"]:
        df = pd.read_csv(f"{D}/{s}/{s}.csv")
        out[s] = np.stack([np.asarray(Image.open(f"{D}/{s}/{q}").convert("RGB")) for q in df["image"]])
        if s == "train":
            out["y"] = df["label"].map(CLASSES.index).values
    np.savez(p, **out)
    return out


# ------------------------------------------------------------------ models
def cbr(i, o, s=1, k=3):
    return [nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


class TinyCNN(nn.Module):
    def __init__(self, w=24, drop=0.2):
        super().__init__()
        self.f = nn.Sequential(*cbr(3, w, 2, 5), *cbr(w, w), nn.MaxPool2d(2),
                               *cbr(w, 2 * w), *cbr(2 * w, 2 * w), nn.MaxPool2d(2),
                               *cbr(2 * w, 4 * w), *cbr(4 * w, 4 * w), nn.MaxPool2d(2), *cbr(4 * w, 8 * w))
        self.h = nn.Sequential(nn.Dropout(drop), nn.Linear(16 * w, 5))

    def forward(self, x):
        f = self.f(x)
        return self.h(torch.cat([Fn.adaptive_avg_pool2d(f, 1).flatten(1), Fn.adaptive_max_pool2d(f, 1).flatten(1)], 1))


class GemHead(nn.Module):
    def __init__(self, c, drop=0.2):
        super().__init__()
        self.h = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * c, 5))

    def forward(self, f):
        return self.h(torch.cat([Fn.adaptive_avg_pool2d(f, 1).flatten(1), Fn.adaptive_max_pool2d(f, 1).flatten(1)], 1))


class TVNet(nn.Module):
    """torchvision backbone (pretrained) + avg/max head."""

    def __init__(self, arch: str):
        super().__init__()
        import torchvision.models as tvm
        nomp = arch.endswith("nm")
        base = arch[:-2] if nomp else arch
        if base in ("r18", "r34", "r50"):
            name = {"r18": "resnet18", "r34": "resnet34", "r50": "resnet50"}[base]
            m = getattr(tvm, name)(weights="DEFAULT")
            layers = [m.conv1, m.bn1, m.relu] + ([] if nomp else [m.maxpool]) + [m.layer1, m.layer2, m.layer3, m.layer4]
            self.body = nn.Sequential(*layers)
            c = m.fc.in_features
        elif base == "effb0":
            m = tvm.efficientnet_b0(weights="DEFAULT")
            self.body = m.features
            c = 1280
        elif base == "mnv3":
            m = tvm.mobilenet_v3_large(weights="DEFAULT")
            self.body = m.features
            c = 960
        elif base == "rgy8":
            m = tvm.regnet_y_800mf(weights="DEFAULT")
            self.body = nn.Sequential(m.stem, m.trunk_output)
            c = 784
        elif base == "cnxt":
            m = tvm.convnext_tiny(weights="DEFAULT")
            self.body = m.features
            c = 768
        elif base == "dn121":
            m = tvm.densenet121(weights="DEFAULT")
            self.body = nn.Sequential(m.features, nn.ReLU(inplace=True))
            c = 1024
        else:
            raise ValueError(arch)
        self.head = GemHead(c)

    def forward(self, x):
        return self.head(self.body(x))


def make_model(arch: str) -> nn.Module:
    if arch.startswith("tiny"):
        w = int(arch[4:] or 24)
        return TinyCNN(w)
    return TVNet(arch)


# ------------------------------------------------------------------ aug
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def gauss_kernel(sigma: float) -> torch.Tensor:
    r = max(1, int(3 * sigma + 0.5))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = gauss_kernel(sigma).to(x.device)
    r = (len(k) - 1) // 2
    c = x.shape[1]
    x = Fn.conv2d(Fn.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(c, 1, 1, -1).contiguous(), groups=c)
    x = Fn.conv2d(Fn.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(c, 1, -1, 1).contiguous(), groups=c)
    return x


class Prep:
    def __init__(self, res: int):
        self.res = res
        self.mean, self.std = MEAN.to(DEV), STD.to(DEV)

    def __call__(self, xb: torch.Tensor) -> torch.Tensor:
        x = xb.to(DEV, non_blocking=True).float().div_(255.0)
        return x

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        if self.res != 96:
            x = Fn.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False)
        return x


def augment(x: torch.Tensor, rng: np.random.Generator, cfg: dict) -> torch.Tensor:
    """x in [0,1] on DEV. dihedral, colour jitter, random blur (match test blur: sigma ~1.0-1.6), noise, crop-shift."""
    b = x.shape[0]
    if rng.random() < 0.5:
        x = x.flip(3)
    if rng.random() < 0.5:
        x = x.flip(2)
    x = torch.rot90(x, int(rng.integers(4)), (2, 3))
    # per-sample brightness / colour cast
    x = x * (1 + cfg["bj"] * (torch.rand(b, 1, 1, 1, device=DEV) - 0.5)) + cfg["cj"] * (torch.rand(b, 3, 1, 1, device=DEV) - 0.5)
    # random translation (wrap-free) via roll + small scale
    if cfg["shift"] > 0:
        s = int(rng.integers(-cfg["shift"], cfg["shift"] + 1)), int(rng.integers(-cfg["shift"], cfg["shift"] + 1))
        x = torch.roll(x, s, (2, 3))
    m = torch.rand(b, device=DEV) < cfg["blur_p"]
    if m.any():
        sig = float(rng.uniform(cfg["blur_lo"], cfg["blur_hi"]))
        x[m] = blur(x[m], sig)
    return x


@torch.no_grad()
def predict(model, X: torch.Tensor, prep: Prep, sigma: float = 0.0, tta: int = 8) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), 128):
        x = prep(X[i:i + 128])
        if sigma > 0:
            x = blur(x, sigma)
        x = prep.norm(x)
        views = [x, x.flip(3), x.flip(2), torch.rot90(x, 1, (2, 3)),
                 torch.rot90(x, 2, (2, 3)), torch.rot90(x, 3, (2, 3)), torch.rot90(x.flip(3), 1, (2, 3)),
                 torch.rot90(x.flip(2), 1, (2, 3))][:tta]
        p = sum(model(v).float().softmax(1) for v in views) / len(views)
        out.append(p.cpu())
    return torch.cat(out).numpy()


def soft_ce(out, t):
    return -(t * Fn.log_softmax(out, 1)).sum(1).mean()


def train(arch, X, y, seed, cfg, Xv=None, yv=None, prep=None):
    """y: int labels (N,) or soft targets (N,5)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(arch).to(DEV)
    ep, bs = cfg["epochs"], cfg["bs"]
    pre = arch.startswith("tiny") is False
    params = model.parameters()
    if pre:
        bb = [p for n, p in model.named_parameters() if not n.startswith("head")]
        hd = [p for n, p in model.named_parameters() if n.startswith("head")]
        params = [{"params": bb, "lr": cfg["lr"]}, {"params": hd, "lr": cfg["lr"] * 5}]
        maxlr = [cfg["lr"], cfg["lr"] * 5]
    else:
        maxlr = cfg["lr"]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    steps = ep * (len(X) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=maxlr, total_steps=steps, pct_start=0.15)
    if y.ndim == 1:
        yt = Fn.one_hot(torch.tensor(y), 5).float()
    else:
        yt = torch.tensor(y).float()
    yt = (yt * (1 - cfg["ls"]) + cfg["ls"] / 5).to(DEV)
    for e in range(ep):
        model.train()
        perm = torch.randperm(len(X))
        t0, tl = time.time(), 0.0
        for i in range(0, len(X) - bs + 1, bs):
            idx = perm[i:i + bs]
            x = prep.norm(augment(prep(X[idx]), rng, cfg))
            yb = yt[idx.to(DEV)]
            if cfg["mix"] > 0 and rng.random() < cfg["mix"]:
                lam = float(rng.beta(1.0, 1.0))
                j = torch.randperm(len(idx), device=DEV)
                H = x.shape[2]
                cut = int(H * np.sqrt(1 - lam))
                cy, cx = int(rng.integers(H)), int(rng.integers(H))
                y1, y2, x1, x2 = max(cy - cut // 2, 0), min(cy + cut // 2, H), max(cx - cut // 2, 0), min(cx + cut // 2, H)
                x[:, :, y1:y2, x1:x2] = x[j][:, :, y1:y2, x1:x2]
                lam = 1 - (y2 - y1) * (x2 - x1) / (H * H)
                out = model(x)
                loss = soft_ce(out, lam * yb + (1 - lam) * yb[j])
            else:
                loss = soft_ce(model(x), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
            tl += loss.item()
        msg = f"ep{e} loss={tl / (len(X) // bs):.3f} {time.time() - t0:.0f}s"
        if Xv is not None and (e % 5 == 4 or e == ep - 1):
            from sklearn.metrics import f1_score
            pv = predict(model, Xv, prep, tta=1)
            msg += f" va F1={f1_score(yv, pv.argmax(1), average='macro'):.4f}"
        log(msg)
    return model


def main():
    arch, epochs, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    folds_run = [int(f) for f in (sys.argv[4] if len(sys.argv) > 4 else "0,1,2,3,4").split(",")]
    cfg = dict(epochs=epochs, bs=64, lr=1e-3 if arch.startswith("tiny") else 5e-4, wd=1e-4 if arch.startswith("tiny") else 1e-2,
               ls=0.05, pl="", pl_thr=0.0, pl_T=1.0, bj=0.2, cj=0.2, shift=0, blur_p=0.3, blur_lo=0.8, blur_hi=1.8, mix=0.0, res=96, tag="")
    for kv in sys.argv[5:]:
        k, v = kv.split("=")
        cfg[k] = type(cfg[k])(v) if not isinstance(cfg[k], str) else v
    tag = cfg["tag"] or f"{arch}_e{epochs}_s{seed}"
    os.makedirs(f"{HERE}/out", exist_ok=True)
    c = load_cache()
    y = c["y"]
    Xtr = torch.tensor(c["train"]).permute(0, 3, 1, 2).contiguous()
    Xte = torch.tensor(np.concatenate([c["public_test"], c["private_test"]])).permute(0, 3, 1, 2).contiguous()
    if DEV.type == "cuda":
        Xtr, Xte = Xtr.pin_memory(), Xte.pin_memory()
    prep = Prep(cfg["res"])
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y))
    log(f"{tag} dev={DEV} cfg={cfg}")
    for k in folds_run:
        a, b = folds[k]
        if k == -1:
            pass
        Xa, ya = Xtr[a], y[a]
        if cfg["pl"]:
            # pseudo-labels: soft stack predictions for the test images (co-training from text+tab+image stack)
            P = np.load(f"{HERE}/" + cfg["pl"].replace("FOLD", str(k)))["test"]
            if cfg["pl_T"] != 1.0:
                P = P ** (1 / cfg["pl_T"]); P = P / P.sum(1, keepdims=True)
            keep = P.max(1) >= cfg["pl_thr"]
            Xa = torch.cat([Xa, Xte[torch.tensor(np.where(keep)[0])]])
            ya = np.concatenate([np.eye(5)[ya], P[keep]])
            log(f"pseudo-labels: +{keep.sum()} test images")
        m = train(arch, Xa, ya, seed * 100 + k, cfg, Xtr[b], y[b], prep)
        pv = predict(m, Xtr[b], prep)
        pvb = predict(m, Xtr[b], prep, sigma=1.3)  # synthetic blurred validation copy (for blur gating)
        pt = predict(m, Xte, prep)
        from sklearn.metrics import f1_score
        log(f"{tag} fold{k} TTA F1={f1_score(y[b], pv.argmax(1), average='macro'):.4f} blurredcopy F1={f1_score(y[b], pvb.argmax(1), average='macro'):.4f}")
        np.savez(f"{HERE}/out/{tag}_f{k}.npz", idx=b, oof=pv, oofb=pvb, test=pt)
        if DEV.type == "cuda":
            log(f"max mem {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
        del m
        torch.cuda.empty_cache() if DEV.type == "cuda" else None


if __name__ == "__main__":
    main()
