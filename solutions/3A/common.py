"""Shared pieces for task 3A: paths, mel front-end, cache, mixture synthesis, model."""
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

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_num_threads(int(os.environ.get("THREADS_3A", "2")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":  # shared 4 GB GPU: cap each process
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRAC_3A", "0.28")))

DATA = Path(os.environ.get("DATA_3A", "/home/minh/Desktop/olympiad_ai/warmup/3A_bird_audio/dataset"))
OUT = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("CACHE_3A", "/tmp/claude-1000/-home-minh-Desktop-olympiad-ai/1f52811c-e600-457a-bc8e-50e6365e680a/scratchpad/cache"))
CACHE.mkdir(parents=True, exist_ok=True)

SPECIES = ["blnori1", "colsco1", "grecou1", "grehor1", "greyel", "orimag1",
           "rewbul", "silphe", "spodov", "wcrlau1"]
SR, N_FFT, HOP, N_MELS = 8000, 512, 128, 64
EPS = 1e-6

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, f_min=50, f_max=4000, power=2.0)


def read_csvs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(DATA / "train/train.csv")
    pu = pd.read_csv(DATA / "public_test/public_test.csv")
    pr = pd.read_csv(DATA / "private_test/private_test.csv")
    return tr, pu, pr


@torch.no_grad()
def mel_power(df: pd.DataFrame, split_dir: Path) -> np.ndarray:
    """Linear mel power (N, 64, 251), each clip RMS-normalised so levels are comparable."""
    out = []
    for f in df["filename"]:
        _, x = wavfile.read(split_dir / f)
        x = x.astype(np.float32) / 32768.0
        x = x / (np.sqrt((x ** 2).mean()) + 1e-8) * 0.05
        out.append(_mel(torch.from_numpy(x)).numpy())
    return np.stack(out).astype(np.float32)


def load_cache() -> dict[str, np.ndarray]:
    f = CACHE / "mels.npz"
    if not f.exists():
        tr, pu, pr = read_csvs()
        np.savez(f, train=mel_power(tr, DATA / "train"), public=mel_power(pu, DATA / "public_test"),
                 private=mel_power(pr, DATA / "private_test"))
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


def denoise(p: np.ndarray, q: float = 50, k: float = 1.0) -> np.ndarray:
    """Spectral subtraction of the per-bin time-percentile (removes the focal clip's own background)."""
    floor = np.percentile(p, q, axis=-1, keepdims=True)
    return np.maximum(p - k * floor, 0.0).astype(np.float32)


def stationary_floor(p: np.ndarray, q: float = 20) -> np.ndarray:
    """Per-bin low percentile of a test clip: its stationary background (insects, hum, hiss)."""
    return np.percentile(p, q, axis=-1).astype(np.float32)  # (N, 64)


# ---------------------------------------------------------------- features
def to_features(p: torch.Tensor) -> torch.Tensor:
    """mel power (B, 64, T) -> 2-channel log features: clip-centred log-mel, per-bin median-removed log-mel."""
    lg = torch.log(p + EPS)
    c1 = lg - lg.flatten(1).median(dim=1).values[:, None, None]
    c2 = lg - lg.median(dim=2, keepdim=True).values
    return torch.stack([c1 / 4.0, c2 / 4.0], dim=1)


# ---------------------------------------------------------------- model
def cbr(ci: int, co: int, stride: int = 1) -> list[nn.Module]:
    return [nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True)]


class BirdSED(nn.Module):
    """Light VGG-style CNN on 2-channel log-mel (CPU budget) + attention pooling over time (SED head)."""

    def __init__(self, n_classes: int = len(SPECIES), w: int = 16, drop: float = 0.3, tpool: int = 1):
        super().__init__()
        self.stem = nn.Sequential(nn.BatchNorm2d(2), nn.AvgPool2d((1, tpool)) if tpool > 1 else nn.Identity())
        self.features = nn.Sequential(
            *cbr(2, w, 2), *cbr(w, 2 * w), nn.MaxPool2d(2),               # 32x126 -> 16x63
            *cbr(2 * w, 2 * w), *cbr(2 * w, 4 * w), nn.MaxPool2d(2),       # -> 8x31
            *cbr(4 * w, 4 * w), *cbr(4 * w, 8 * w), nn.MaxPool2d((2, 1)),  # -> 4x31
            *cbr(8 * w, 8 * w))
        self.drop = nn.Dropout(drop)
        self.fc = nn.Conv1d(8 * w, 8 * w, 1)
        self.att = nn.Conv1d(8 * w, n_classes, 1)
        self.cla = nn.Conv1d(8 * w, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(self.stem(x))            # (B, C, F', T')
        h = h.mean(2) + h.amax(2)                   # pool frequency
        h = self.drop(F.relu(self.fc(self.drop(h))))
        frame = self.cla(h)                         # frame-wise logits
        a = torch.softmax(torch.tanh(self.att(h)), dim=-1)
        clip = (a * frame).sum(-1)                  # attention-weighted clip logit
        return 0.5 * (clip + frame.amax(-1))


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    cols = [c for c in range(y.shape[1]) if 0 < (y[:, c] > 0.5).sum() < len(y)]
    return float(np.mean([roc_auc_score(y[:, c] > 0.5, p[:, c]) for c in cols]))
