"""Soundscape-like mixture synthesis in the (linear) mel-power domain.

A mixture = background (a real test-clip stationary floor with random texture, plus
procedural distractors seen in the test soundscapes) + 0..3 denoised focal train clips,
each time-shifted, slightly pitch-shifted and scaled to a random SNR.
"""
from __future__ import annotations

import numpy as np
import torch

from common import SR, T

N_MELS = 64
FROGS = True  # frog-chorus + big-arch distractors (added after inspecting test false positives)
FPS = SR / 128.0


def _centers(n: int) -> np.ndarray:
    hz = 700.0 * (10 ** (np.linspace(2595 * np.log10(1 + 50 / 700), 2595 * np.log10(1 + 4000 / 700), n + 2) / 2595) - 1)
    return hz[1:-1]


MEL_CENTERS = _centers(N_MELS)
WSCALE = 1.0


def configure(n_mels: int) -> None:
    """Set the mel resolution used by the procedural distractors (widths scale with n_mels / 64)."""
    global N_MELS, MEL_CENTERS, WSCALE
    N_MELS, MEL_CENTERS, WSCALE = n_mels, _centers(n_mels), n_mels / 64.0


def hz2bin(f: np.ndarray) -> np.ndarray:
    return np.interp(f, MEL_CENTERS, np.arange(N_MELS))


def _track(rng: np.random.Generator, f_t: np.ndarray, on: np.ndarray, width: float = 0.8) -> np.ndarray:
    """Render a frequency trajectory f_t (T,) gated by on (T,) as a gaussian ridge in mel bins."""
    b = hz2bin(f_t)[None, :]
    width = width * WSCALE
    return np.exp(-0.5 * ((np.arange(N_MELS)[:, None] - b) / width) ** 2) * on[None, :]


def distractor(rng: np.random.Generator) -> np.ndarray:
    """One random non-target sound (64, T), unit-ish peak."""
    kind = rng.choice(8, p=[0.12, 0.2, 0.08, 0.08, 0.1, 0.07, 0.25, 0.1]) if FROGS else rng.integers(6)
    t = np.arange(T) / FPS
    out = np.zeros((N_MELS, T), np.float32)
    if kind == 0:  # pulsed insect tone at 3-3.9 kHz
        f = rng.uniform(3000, 3900)
        per, duty, ph = rng.uniform(0.12, 0.45), rng.uniform(0.3, 0.7), rng.uniform(0, 1)
        on = (((t / per) + ph) % 1 < duty).astype(float)
        out += _track(rng, np.full(T, f), on, rng.uniform(0.6, 1.5))
    elif kind == 7:  # large harmonic arch (seen in test): f0 ~0.6-1.2 kHz rising 0.3-0.9 kHz, 2-3 harmonics
        for _ in range(rng.integers(1, 3)):
            t0, dur = rng.uniform(-0.3, 3.3), rng.uniform(0.6, 1.6)
            u = (t - t0) / dur
            on = ((u >= 0) & (u <= 1)).astype(float)
            f0 = rng.uniform(600, 1200) + rng.uniform(300, 900) * np.clip(1 - (2 * u - 1) ** 2, 0, 1)
            for h in range(1, rng.integers(2, 4) + 1):
                out += _track(rng, f0 * h, on, 1.2) * rng.uniform(0.5, 1.0)
    elif kind == 6:  # frog / insect chorus: regular pulse train over most of the clip, 1-3 harmonic dots
        for _ in range(rng.integers(1, 3)):
            per = rng.uniform(0.08, 0.3)
            t0, t1 = (0.0, 4.0) if rng.random() < 0.7 else sorted(rng.uniform(0, 4, 2))
            f0, nh = rng.uniform(350, 900), rng.integers(1, 4)
            plen, slope = rng.uniform(0.02, 0.06), rng.uniform(-0.3, 0.6)
            ph = rng.uniform(0, per)
            u = ((t - ph) % per) / plen
            on = ((u < 1) & (t >= t0) & (t <= t1)).astype(float) * (1 + rng.normal(0, 0.15, T)).clip(0.3)
            for h in range(1, nh + 1):
                out += _track(rng, f0 * h * (1 + slope * np.clip(u, 0, 1)), on, 0.8) * rng.uniform(0.4, 1.0)
    elif kind == 1:  # harmonic arch / sweep (non-target bird or frog)
        for _ in range(rng.integers(1, 3)):
            t0, dur = rng.uniform(-0.3, 3.5), rng.uniform(0.3, 1.6)
            u = (t - t0) / dur
            on = ((u >= 0) & (u <= 1)).astype(float)
            f0 = rng.uniform(500, 1400) + rng.uniform(-1, 1) * 500 * (1 - (2 * u - 1) ** 2) * rng.choice([1, -1])
            for h in range(1, rng.integers(2, 4) + 1):
                out += _track(rng, f0 * h, on, 0.9) / h
    elif kind == 2:  # engine / generator hum harmonics
        f0 = rng.uniform(50, 220)
        t0, t1 = sorted(rng.uniform(0, 4, 2)) if rng.random() < 0.5 else (0, 4)
        on = ((t >= t0) & (t <= t1)).astype(float)
        for h in range(1, int(1800 / f0)):
            out += _track(rng, np.full(T, f0 * h), on, 0.6) * rng.uniform(0.2, 1.0)
    elif kind == 3:  # rain: broadband impulses + hiss
        drops = (rng.random(T) < rng.uniform(0.05, 0.4)).astype(float) * rng.exponential(1, T)
        out += drops[None, :] * np.linspace(1.0, rng.uniform(0.2, 1.5), N_MELS)[:, None]
        out += rng.uniform(0.05, 0.3)
    elif kind == 4:  # cicada band, amplitude-modulated
        c, w = rng.uniform(1800, 3800), rng.uniform(150, 700)
        band = np.exp(-0.5 * ((MEL_CENTERS - c) / w) ** 2)[:, None]
        am = 1 + rng.uniform(0, 1) * np.sin(2 * np.pi * rng.uniform(0.2, 12) * t + rng.uniform(0, 6))
        env = np.clip(np.cumsum(rng.normal(0, 0.1, T)) + 1, 0.2, 2) if rng.random() < 0.5 else 1.0
        out += band * (am * env)[None, :]
    else:  # broadband click train / wind gusts (vertical streaks, low-mid)
        on = (rng.random(T) < rng.uniform(0.02, 0.1)).astype(float)
        out += on[None, :] * np.exp(-MEL_CENTERS / rng.uniform(500, 3000))[:, None]
    return out.astype(np.float32)


class Mixer:
    def __init__(self, birds: np.ndarray, labels: np.ndarray, floors: np.ndarray, seed: int = 0,
                 snr_db: tuple[float, float] = (-12.0, 12.0), p_k: tuple[float, ...] = (0.15, 0.4, 0.3, 0.15),
                 n_distract: float = 1.5, pitch: int = 2, test_bgs: np.ndarray | None = None,
                 test_soft: np.ndarray | None = None, p_test_bg: float = 0.0, balance: float = 0.5,
                 eq: float = 0.0, p_gate: float = 0.0):
        self.birds, self.labels, self.floors = birds, labels, floors
        self.rng = np.random.default_rng(seed)
        self.snr, self.p_k, self.n_distract, self.pitch = snr_db, np.array(p_k), n_distract, pitch
        self.test_bgs, self.test_soft, self.p_test_bg = test_bgs, test_soft, p_test_bg
        cnt = labels.sum(0)
        w = (labels / np.maximum(cnt, 1) ** balance).sum(1)
        self.w = w / w.sum()
        self.energy = birds.mean(axis=(1, 2))
        self.eq, self.p_gate = eq, p_gate

    def background(self) -> tuple[np.ndarray, np.ndarray]:
        r = self.rng
        y = np.zeros(self.labels.shape[1], np.float32)
        if self.test_bgs is not None and r.random() < self.p_test_bg:
            i = r.integers(len(self.test_bgs))
            bg = np.roll(self.test_bgs[i], r.integers(T), axis=1).copy()
            y = self.test_soft[i].copy()
        else:
            fl = self.floors[r.integers(len(self.floors))]
            tex = r.gamma(r.uniform(1, 4), 1.0, (64, T)).astype(np.float32)  # drawn at 64 bins: same RNG stream for any n_mels
            if N_MELS != 64:
                tex = tex[np.minimum((np.arange(N_MELS) * 64) // N_MELS, 63)]
            tex /= tex.mean()
            slow = np.exp(np.cumsum(r.normal(0, 0.03, T)))[None, :]
            bg = fl[:, None] * tex * slow / slow.mean()
        lvl = bg.mean()
        for _ in range(r.poisson(self.n_distract)):
            d = distractor(r)
            bg = bg + d * lvl * 10 ** (r.uniform(-5, 15) / 10) / (d.mean() + 1e-6) * 0.1
        return bg.astype(np.float32), y

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        r = self.rng
        bg, y = self.background()
        k = r.choice(len(self.p_k), p=self.p_k / self.p_k.sum())
        mix = bg.copy()
        lvl = bg.mean()
        for i in r.choice(len(self.birds), size=k, replace=False, p=self.w):
            b = np.roll(self.birds[i], r.integers(T), axis=1).astype(np.float32)
            s = r.integers(-self.pitch, self.pitch + 1)
            if s:
                b = np.roll(b, s, axis=0)
                if s > 0:
                    b[:s] = 0
                else:
                    b[s:] = 0
            if self.eq > 0:  # random smooth spectral tilt/bump (recorder + distance filtering)
                z = np.linspace(-1, 1, b.shape[0])
                tilt = r.normal(0, self.eq) * z + r.normal(0, self.eq / 2) * np.cos(np.pi * (z * r.uniform(1, 3) + r.uniform(0, 2)))
                b = b * np.exp(tilt)[:, None].astype(np.float32)
            if self.p_gate > 0 and r.random() < self.p_gate:  # bird only calls in part of the window
                L = r.integers(T // 3, T)
                st = r.integers(0, T - L + 1)
                g8 = np.zeros(T, np.float32)
                g8[st:st + L] = 1
                b = b * g8[None, :]
            g = lvl * 10 ** (r.uniform(*self.snr) / 10) / (self.energy[i] + 1e-9)
            mix += g * b
            y = np.maximum(y, self.labels[i])
        return mix, y

    def batch(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = zip(*(self.sample() for _ in range(n)))
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))
