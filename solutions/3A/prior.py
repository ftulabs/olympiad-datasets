"""Metadata prior: species activity by hour of day, estimated from train recording hours."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import SPECIES


def hour_activity(tr: pd.DataFrame, window: int = 1, alpha: float = 0.5) -> np.ndarray:
    """(24, n_species) activity in (0, 1]: circularly smoothed hour counts / per-species max."""
    cnt = np.zeros((24, len(SPECIES)))
    for h, s in zip(tr["hour"], tr["primary_label"]):
        cnt[h, SPECIES.index(s)] += 1
    sm = sum(np.roll(cnt, k, axis=0) for k in range(-window, window + 1))
    return (sm + alpha) / (sm.max(0, keepdims=True) + alpha)


def apply_prior(p: np.ndarray, hours: np.ndarray, act: np.ndarray, beta: float = 0.5) -> np.ndarray:
    return p * act[hours] ** beta
