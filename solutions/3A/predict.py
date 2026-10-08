"""Ensemble checkpoints with time-shift TTA -> submissions (+ optional pseudo-labels for round 2).

usage: python predict.py CKPT_NAME [CKPT_NAME ...] [--tta 4] [--pseudo_out FILE] [--no_csv]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from common import CACHE, DEVICE, OUT, SPECIES, BirdSED, load_cache, read_csvs
from train import predict


def ensemble(names: list[str], P: torch.Tensor, tta: int) -> np.ndarray:
    preds = []
    for n in names:
        sd = torch.load(CACHE / f"{n}.pt", map_location="cpu")
        sd = {("stem.0." + k[5:] if k.startswith("stem.") and not k.startswith("stem.0") else k): v for k, v in sd.items()}
        cfg_f = CACHE / f"{n.removesuffix('_last')}.json"
        cfg = json.loads(cfg_f.read_text()) if cfg_f.exists() else {}
        m = BirdSED(w=sd["features.0.weight"].shape[0], tpool=cfg.get("tpool", 1))
        m.load_state_dict(sd)
        m.to(DEVICE)
        preds.append(predict(m, P, tta=tta))
    return np.mean(preds, axis=0)


def write(df: pd.DataFrame, probs: np.ndarray, path: str) -> None:
    sub = pd.DataFrame(probs, columns=SPECIES)
    sub.insert(0, "id", df["id"].values)
    sub.to_csv(path, index=False)
    print("wrote", path, sub.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--pseudo_out", default="")
    ap.add_argument("--no_csv", action="store_true")
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    d = load_cache()
    _, pu, pr = read_csvs()
    P = torch.from_numpy(np.concatenate([d["public"], d["private"]]))
    probs = ensemble(a.names, P, a.tta)
    if a.pseudo_out:
        np.savez(a.pseudo_out, soft=np.where(probs > 0.1, probs, 0.0))
        print("pseudo labels: mean #species/clip =", (probs > 0.5).sum(1).mean())
    if not a.no_csv:
        write(pu, probs[: len(pu)], str(OUT / f"{a.prefix}public_submission.csv"))
        write(pr, probs[len(pu):], str(OUT / f"{a.prefix}private_submission.csv"))
    np.save(CACHE / f"probs_{'_'.join(a.names)[:80]}.npy", probs)


if __name__ == "__main__":
    main()
