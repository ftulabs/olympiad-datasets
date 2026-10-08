"""Tune blend weights and the negative-binomial median post-processing on out-of-fold backtests.

python tune.py oof_dir V25:mlp_a,glm_a V24:mlp_a,glm_a
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
from scipy.stats import nbinom, poisson


def wape(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * np.abs(y - p)) / np.sum(w * y))


def nb_q(mu: np.ndarray, phi: float, q: float) -> np.ndarray:
    mu = np.maximum(mu, 1e-6)
    if phi <= 0:
        return poisson.ppf(q, mu)
    r = 1 / phi
    return nbinom.ppf(q, r, r / (r + mu))


def load(d: str, fold: str, name: str) -> dict:
    z = np.load(f"{d}/{fold}_{name}.npz")
    return {k: z[k] for k in z.files}


def main() -> None:
    d = sys.argv[1]
    specs = [a.split(":") for a in sys.argv[2:]]
    folds = {f: [load(d, f, n) for n in names.split(",")] for f, names in specs}
    nmod = len(next(iter(folds.values())))
    print("single models:")
    for f, zs in folds.items():
        print(f, [round(wape(*(lambda m: (z["y"][m], z["pred"][m], z["w"][m]))(~np.isnan(z["y"]) & ~z["so"])), 4)
                  for z in zs])
    grid = [w for w in itertools.product(*[np.linspace(0, 1, 6)] * nmod) if abs(sum(w) - 1) < 1e-9]
    res = []
    for wts in grid:
        sc = []
        for f, zs in folds.items():
            m = ~np.isnan(zs[0]["y"]) & ~zs[0]["so"]
            p = np.exp(sum(w * np.log(np.maximum(z["pred"], 1e-6)) for w, z in zip(wts, zs)))
            sc.append(wape(zs[0]["y"][m], p[m], zs[0]["w"][m]))
        res.append((np.mean(sc), wts, sc))
    res.sort(key=lambda r: r[0])
    for r in res[:5]:
        print("blend", np.round(r[1], 2), round(r[0], 4), np.round(r[2], 4))
    wts = res[0][1]
    print("median post-processing on best blend:")
    for phi, q, c in itertools.product([0, 0.05, 0.1, 0.2, 0.3], [0.45, 0.5, 0.55], [1.0]):
        sc = []
        for f, zs in folds.items():
            m = ~np.isnan(zs[0]["y"]) & ~zs[0]["so"]
            p = np.exp(sum(w * np.log(np.maximum(z["pred"], 1e-6)) for w, z in zip(wts, zs)))
            sc.append(wape(zs[0]["y"][m], nb_q(p[m] * c, phi, q), zs[0]["w"][m]))
        print(f"phi={phi} q={q} -> {np.mean(sc):.4f} {np.round(sc, 4)}")
    for c in (0.9, 0.95, 1.0, 1.05):
        sc = []
        for f, zs in folds.items():
            m = ~np.isnan(zs[0]["y"]) & ~zs[0]["so"]
            p = np.exp(sum(w * np.log(np.maximum(z["pred"], 1e-6)) for w, z in zip(wts, zs)))
            sc.append(wape(zs[0]["y"][m], p[m] * c, zs[0]["w"][m]))
        print(f"scale {c}: {np.mean(sc):.4f} {np.round(sc, 4)}")


if __name__ == "__main__":
    main()
