"""Ensemble trained models -> submissions (+ soft pseudo-labels for the next self-training round).

usage: python predict.py NAME [NAME ...] [--weights w1 w2 ...] [--rank] [--pseudo_out FILE] [--no_csv] [--recompute]
Each NAME refers to CACHE/NAME.pt + NAME.json; by default the TTA test probabilities written at the end of
training (CACHE/NAME_test.npy) are used, --recompute re-runs inference from the checkpoint.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from common import CACHE, DEVICE, HERE, SPECIES, build_model, load_cache, read_csvs


def model_probs(name: str, recompute: bool, tta: int = 4) -> np.ndarray:
    f = CACHE / f"{name}_test.npy"
    if f.exists() and not recompute:
        return np.load(f)
    from train import predict
    cfg = json.loads((CACHE / f"{name}.json").read_text())["cfg"]
    m = build_model({**cfg, "pretrained": False})
    m.load_state_dict(torch.load(CACHE / f"{name}.pt", map_location="cpu"))
    m.to(DEVICE)
    d = load_cache(cfg["mels"], cfg["nfft"])
    P = torch.from_numpy(np.concatenate([d["public"], d["private"]]))
    p = predict(m, P, cfg["nch"], tta=tta)
    np.save(f, p)
    return p


def blend(probs: list[np.ndarray], weights: list[float], rank: bool) -> np.ndarray:
    w = np.asarray(weights, float) / np.sum(weights)
    if rank:  # per-species rank average (AUC only depends on ranks); rescaled to [0, 1]
        r = [np.column_stack([rankdata(p[:, c]) / len(p) for c in range(p.shape[1])]) for p in probs]
        return np.tensordot(w, np.stack(r), axes=1)
    return np.tensordot(w, np.stack(probs), axes=1)


def write(df: pd.DataFrame, probs: np.ndarray, path: str) -> None:
    sub = pd.DataFrame(probs, columns=SPECIES)
    sub.insert(0, "id", df["id"].values)
    sub.to_csv(path, index=False)
    print("wrote", path, sub.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--weights", type=float, nargs="*")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--pseudo_out", default="")
    ap.add_argument("--no_csv", action="store_true")
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    _, pu, pr = read_csvs()
    probs = [model_probs(n, a.recompute) for n in a.names]
    p = blend(probs, a.weights or [1.0] * len(probs), a.rank)
    if a.pseudo_out:  # pseudo-labels are always the probability mean (not ranks)
        soft = blend(probs, a.weights or [1.0] * len(probs), False)
        np.save(a.pseudo_out, soft.astype(np.float32))
        print("pseudo labels: mean #species/clip (p>0.5) =", (soft > 0.5).sum(1).mean())
    if not a.no_csv:
        write(pu, p[: len(pu)], f"{a.out}/public_submission.csv")
        write(pr, p[len(pu):], f"{a.out}/private_submission.csv")


if __name__ == "__main__":
    main()
