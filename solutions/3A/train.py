"""Train one BirdSED model on synthetic soundscape mixtures; log soundscape-like validation AUC.

usage: python train.py --name NAME [--epochs 30] [--seed 0] [--full] [--pseudo FILE] ...
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from common import CACHE, DEVICE, BirdSED, denoise, load_cache, macro_auc, read_csvs, stationary_floor, to_features, train_targets
from mix import Mixer


def spec_augment(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    x = x.clone()
    for i in range(len(x)):
        for _ in range(2):
            f0, fw = rng.integers(0, 64), rng.integers(0, 8)
            x[i, :, f0:f0 + fw] = 0
            t0, tw = rng.integers(0, 251), rng.integers(0, 25)
            x[i, :, :, t0:t0 + tw] = 0
    return x


@torch.no_grad()
def predict(model: nn.Module, P: torch.Tensor, bs: int = 50, tta: int = 0) -> np.ndarray:
    model.eval()
    outs = []
    shifts = [0] + [int(s) for s in np.linspace(0, 251, tta + 2)[1:-1]]
    for i in range(0, len(P), bs):
        xb = P[i:i + bs].to(DEVICE)
        logit = sum(model(to_features(torch.roll(xb, s, dims=-1))) for s in shifts) / len(shifts)
        outs.append(torch.sigmoid(logit).cpu())
    return torch.cat(outs).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--per_epoch", type=int, default=1600)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--tpool", type=int, default=1, help="extra time pooling after the stem (speed)")
    ap.add_argument("--full", action="store_true", help="train on all train clips (no hold-out)")
    ap.add_argument("--no_denoise", action="store_true")
    ap.add_argument("--no_distract", action="store_true")
    ap.add_argument("--no_specaug", action="store_true")
    ap.add_argument("--plain_bg", action="store_true", help="white-noise backgrounds instead of test floors")
    ap.add_argument("--sec", type=float, default=1.0)
    ap.add_argument("--pitch", type=int, default=2)
    ap.add_argument("--snr_lo", type=float, default=-10)
    ap.add_argument("--loss", default="bce", choices=["bce", "focal"])
    ap.add_argument("--pseudo", default="", help="npz with test pseudo-labels -> use test clips as backgrounds")
    ap.add_argument("--p_test_bg", type=float, default=0.5)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    d = load_cache()
    tr, _, _ = read_csvs()
    y = train_targets(tr, a.sec)
    dn = denoise(d["train"])
    birds = d["train"] if a.no_denoise else dn
    test_all = np.concatenate([d.pop("public"), d.pop("private")])
    vfl = stationary_floor(test_all)
    floors = vfl.copy()
    if a.plain_bg:
        floors = np.ones_like(floors) * floors.mean()

    # stratified hold-out of 20% focal clips -> validation mixtures never share a bird clip with training
    vrng = np.random.default_rng(123)
    val_idx = np.concatenate([vrng.permutation(np.where(tr.primary_label == s)[0])[: max(1, len(np.where(tr.primary_label == s)[0]) // 5)]
                              for s in sorted(tr.primary_label.unique())])
    tr_idx = np.setdiff1d(np.arange(len(tr)), val_idx)
    if a.full:
        tr_idx = np.arange(len(tr))

    # validation: fixed soundscape-like mixtures of held-out clips over held-out test floors (always same recipe)
    vm = Mixer(dn[val_idx], train_targets(tr, 1.0)[val_idx], vfl[1::2], seed=999)
    Xv, Yv = vm.batch(800)

    kw = {}
    if a.pseudo:
        ps = np.load(a.pseudo)
        kw = dict(test_bgs=test_all, test_soft=ps["soft"].astype(np.float32), p_test_bg=a.p_test_bg)
    if not a.pseudo:
        del test_all
    m = Mixer(birds[tr_idx], y[tr_idx], floors[0::2] if not a.full else floors, seed=a.seed + 1,
              snr_db=(a.snr_lo, 12.0), n_distract=0.0 if a.no_distract else 1.5, pitch=a.pitch, **kw)

    model = BirdSED(w=a.width, tpool=a.tpool).to(DEVICE)
    (CACHE / f"{a.name}.json").write_text(json.dumps(dict(width=a.width, tpool=a.tpool)))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    steps = a.epochs * (a.per_epoch // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def loss_fn(logit: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        l = bce(logit, t)
        if a.loss == "focal":
            p = torch.sigmoid(logit)
            pt = p * t + (1 - p) * (1 - t)
            l = l * (1 - pt) ** 2 * 4
        return l.mean()

    log, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for _ in range(a.per_epoch // a.bs):
            xb, yb = m.batch(a.bs)
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            feats = to_features(xb)
            if not a.no_specaug:
                feats = spec_augment(feats, rng)
            loss = loss_fn(model(feats), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        auc = macro_auc(Yv.numpy(), predict(model, Xv))
        log.append(dict(ep=ep, loss=tot / (a.per_epoch // a.bs), val_auc=auc, sec=time.time() - t0))
        print(json.dumps(log[-1]), flush=True)
        if ep > a.epochs // 2 and auc >= best:
            best = auc
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, CACHE / f"{a.name}.pt")
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, CACHE / f"{a.name}_last.pt")
    print("BEST", a.name, best, "LAST", log[-1]["val_auc"], flush=True)


if __name__ == "__main__":
    main()
