"""Shared pieces for task 3A v2: paths, mel front-end + cache, features, models, metric."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from scipy.io import wavfile
from sklearn.metrics import roc_auc_score

torch.set_num_threads(int(os.environ.get("THREADS_3A", "2")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda" and os.environ.get("GPU_FRAC_3A"):  # shared GPU: cap this process
    torch.cuda.set_per_process_memory_fraction(float(os.environ["GPU_FRAC_3A"]))

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("DATA_3A", "/home/minh/Desktop/olympiad_ai/warmup/3A_bird_audio/dataset"))
CACHE = Path(os.environ.get("CACHE_3A", str(HERE / ".cache")))
CACHE.mkdir(parents=True, exist_ok=True)

SPECIES = ["blnori1", "colsco1", "grecou1", "grehor1", "greyel", "orimag1",
           "rewbul", "silphe", "spodov", "wcrlau1"]
SR, HOP, T = 8000, 128, 251
EPS = 1e-6


def read_csvs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(DATA / "train/train.csv")
    pu = pd.read_csv(DATA / "public_test/public_test.csv")
    pr = pd.read_csv(DATA / "private_test/private_test.csv")
    return tr, pu, pr


@torch.no_grad()
def mel_power(df: pd.DataFrame, split_dir: Path, n_mels: int, n_fft: int) -> np.ndarray:
    """Linear mel power (N, n_mels, 251); each clip RMS-normalised so levels are comparable."""
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=n_fft, hop_length=HOP, n_mels=n_mels,
                                               f_min=50, f_max=4000, power=2.0)
    out = []
    for f in df["filename"]:
        _, x = wavfile.read(split_dir / f)
        x = x.astype(np.float32) / 32768.0
        x = x / (np.sqrt((x ** 2).mean()) + 1e-8) * 0.05
        out.append(mel(torch.from_numpy(x)).numpy()[:, :T])
    return np.stack(out).astype(np.float32)


def load_cache(n_mels: int = 64, n_fft: int = 512) -> dict[str, np.ndarray]:
    f = CACHE / f"mels_{n_mels}_{n_fft}.npz"
    if not f.exists():
        tr, pu, pr = read_csvs()
        np.savez(f, train=mel_power(tr, DATA / "train", n_mels, n_fft),
                 public=mel_power(pu, DATA / "public_test", n_mels, n_fft),
                 private=mel_power(pr, DATA / "private_test", n_mels, n_fft))
    d = np.load(f)
    return {k: d[k] for k in d.files}


def train_targets(tr: pd.DataFrame, sec_weight: float = 1.0) -> np.ndarray:
    y = np.zeros((len(tr), len(SPECIES)), np.float32)
    for i, (p, s) in enumerate(zip(tr["primary_label"], tr["secondary_labels"].fillna(""))):
        y[i, SPECIES.index(p)] = 1.0
        for t in s.split():
            if t in SPECIES:
                y[i, SPECIES.index(t)] = max(y[i, SPECIES.index(t)], sec_weight)
    return y


def holdout_split(tr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Same stratified 20 % focal hold-out as v1 (seed 123)."""
    vrng = np.random.default_rng(123)
    val_idx = np.concatenate([vrng.permutation(np.where(tr.primary_label == s)[0])[: max(1, (tr.primary_label == s).sum() // 5)]
                              for s in sorted(tr.primary_label.unique())])
    return np.setdiff1d(np.arange(len(tr)), val_idx), val_idx


def denoise(p: np.ndarray, q: float = 50, k: float = 1.0) -> np.ndarray:
    floor = np.percentile(p, q, axis=-1, keepdims=True)
    return np.maximum(p - k * floor, 0.0).astype(np.float32)


def stationary_floor(p: np.ndarray, q: float = 20) -> np.ndarray:
    return np.percentile(p, q, axis=-1).astype(np.float32)


# ---------------------------------------------------------------- features
def to_features(p: torch.Tensor, nch: int = 2) -> torch.Tensor:
    """mel power (B, M, T) -> log features: clip-median-centred, per-bin-median-removed (+ PCEN-like 3rd)."""
    lg = torch.log(p + EPS)
    c1 = lg - lg.flatten(1).median(dim=1).values[:, None, None]
    c2 = lg - lg.median(dim=2, keepdim=True).values
    chans = [c1 / 4.0, c2 / 4.0]
    if nch == 3:  # per-bin 80th-pct removed (stronger stationary-noise suppression), clipped at 0
        q = torch.quantile(lg, 0.8, dim=2, keepdim=True)
        chans.append(torch.clamp(lg - q, min=0) / 2.0)
    return torch.stack(chans, dim=1)


# ---------------------------------------------------------------- models
def cbr(ci: int, co: int, stride: int = 1) -> list[nn.Module]:
    return [nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True)]


class SEDHead(nn.Module):
    """Frame-wise classifier + attention pooling over time; output = mean(attention clip logit, max frame logit)."""

    def __init__(self, c: int, n_classes: int, drop: float):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.fc = nn.Conv1d(c, c, 1)
        self.att = nn.Conv1d(c, n_classes, 1)
        self.cla = nn.Conv1d(c, n_classes, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (B, C, T')
        h = self.drop(F.relu(self.fc(self.drop(h))))
        frame = self.cla(h)
        a = torch.softmax(torch.tanh(self.att(h)), dim=-1)
        return 0.5 * ((a * frame).sum(-1) + frame.amax(-1))


class BirdSED(nn.Module):
    """v1 light VGG-style CNN (kept as one ensemble family)."""

    def __init__(self, n_classes: int = len(SPECIES), w: int = 32, drop: float = 0.3, nch: int = 2, **_: object):
        super().__init__()
        self.stem = nn.BatchNorm2d(nch)
        self.features = nn.Sequential(
            *cbr(nch, w, 2), *cbr(w, 2 * w), nn.MaxPool2d(2),
            *cbr(2 * w, 2 * w), *cbr(2 * w, 4 * w), nn.MaxPool2d(2),
            *cbr(4 * w, 4 * w), *cbr(4 * w, 8 * w), nn.MaxPool2d((2, 1)),
            *cbr(8 * w, 8 * w))
        self.head = SEDHead(8 * w, n_classes, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(self.stem(x))
        return self.head(h.mean(2) + h.amax(2))


class TVSED(nn.Module):
    """torchvision ImageNet backbone (ResNet / EfficientNet / RegNet) on log-mel, frequency pooled, SED head."""

    def __init__(self, arch: str = "resnet18", n_classes: int = len(SPECIES), drop: float = 0.3, nch: int = 3,
                 pretrained: bool = True, **_: object):
        super().__init__()
        import torchvision.models as tvm
        weights = "DEFAULT" if pretrained else None
        net = getattr(tvm, arch)(weights=weights)
        self.norm = nn.BatchNorm2d(nch)
        self.nch = nch
        if arch.startswith("resnet"):
            # keep time resolution higher: stride-1 maxpool removed -> T'/16
            self.body = nn.Sequential(net.conv1, net.bn1, net.relu, net.layer1, net.layer2, net.layer3, net.layer4)
            c = net.fc.in_features
        elif arch.startswith("efficientnet") or arch.startswith("mobilenet"):
            self.body = net.features
            c = net.classifier[-1].in_features
        elif arch.startswith("regnet"):
            self.body = nn.Sequential(net.stem, net.trunk_output)
            c = net.fc.in_features
        elif arch.startswith("densenet"):
            self.body = nn.Sequential(net.features, nn.ReLU(inplace=True))
            c = net.classifier.in_features
        else:
            raise ValueError(arch)
        self.head = SEDHead(c, n_classes, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        if self.nch != 3:
            x = x[:, [0, 1, 1]] if self.nch == 2 else x
        h = self.body(x)  # (B, C, F', T')
        return self.head(h.mean(2) + h.amax(2))


def build_model(cfg: dict) -> nn.Module:
    if cfg.get("arch", "sed") == "sed":
        return BirdSED(w=cfg.get("width", 32), nch=cfg.get("nch", 2), drop=cfg.get("drop", 0.3))
    return TVSED(arch=cfg["arch"], nch=cfg.get("nch", 3), drop=cfg.get("drop", 0.3), pretrained=cfg.get("pretrained", True))


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    cols = [c for c in range(y.shape[1]) if 0 < (y[:, c] > 0.5).sum() < len(y)]
    return float(np.mean([roc_auc_score(y[:, c] > 0.5, p[:, c]) for c in cols]))
