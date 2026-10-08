"""Train one model on synthetic soundscape mixtures; log two soundscape-like validation AUCs; save test predictions.

usage: python train.py --name NAME [--arch sed|resnet18|efficientnet_b0|...] [--mels 64|128] [--full] [--pseudo FILE] ...
Outputs (in CACHE): NAME.pt (EMA weights), NAME.json (config + log), NAME_test.npy (800x10 TTA probs),
NAME_val.npz (SV / RV predictions, for blending studies).
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

import mix
from common import (CACHE, DEVICE, T, build_model, denoise, holdout_split, load_cache, macro_auc, read_csvs,
                    stationary_floor, to_features, train_targets)


class MixStream(IterableDataset):
    """Endless stream of freshly synthesised mixtures; every worker gets its own RNG stream."""

    def __init__(self, n_mels: int, seed: int, kw: dict):
        self.n_mels, self.seed, self.kw = n_mels, seed, kw

    def __iter__(self):
        wi = get_worker_info()
        wid = wi.id if wi else 0
        mix.configure(self.n_mels)
        m = mix.Mixer(seed=self.seed * 100 + wid, **self.kw)
        while True:
            x, y = m.sample()
            yield torch.from_numpy(x), torch.from_numpy(y)


def spec_augment(x: torch.Tensor, rng: np.random.Generator, n_mels: int) -> torch.Tensor:
    x = x.clone()
    fw_max = max(8, n_mels // 8)
    for i in range(len(x)):
        for _ in range(2):
            f0, fw = rng.integers(0, n_mels), rng.integers(0, fw_max)
            x[i, :, f0:f0 + fw] = 0
            t0, tw = rng.integers(0, T), rng.integers(0, 25)
            x[i, :, :, t0:t0 + tw] = 0
    return x


@torch.no_grad()
def predict(model: nn.Module, P: torch.Tensor, nch: int, bs: int = 32, tta: int = 0) -> np.ndarray:
    model.eval()
    outs = []
    shifts = [0] + [int(s) for s in np.linspace(0, T, tta + 2)[1:-1]]
    for i in range(0, len(P), bs):
        xb = P[i:i + bs].to(DEVICE)
        logit = sum(model(to_features(torch.roll(xb, s, dims=-1), nch)) for s in shifts) / len(shifts)
        outs.append(torch.sigmoid(logit).cpu())
    return torch.cat(outs).numpy()


def make_vals(d: dict, tr, n_mels: int, n_fft: int, low_idx: np.ndarray) -> dict[str, np.ndarray]:
    """SV: v1 recipe (held-out birds over odd test floors + distractors). RV: held-out birds over REAL low-activity
    test clips (real insects / frogs / rain / non-target birds), labels = inserted birds only. Cached per front-end."""
    f = CACHE / f"vals_{n_mels}_{n_fft}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    mix.configure(n_mels)
    _, val_idx = holdout_split(tr)
    dn = denoise(d["train"])
    y1 = train_targets(tr, 1.0)
    test_all = np.concatenate([d["public"], d["private"]])
    vfl = stationary_floor(test_all)
    sv = mix.Mixer(dn[val_idx], y1[val_idx], vfl[1::2], seed=999)
    Xs, Ys = sv.batch(1600)
    rv = mix.Mixer(dn[val_idx], y1[val_idx], vfl, seed=777, n_distract=0.0, p_k=(0.1, 0.45, 0.3, 0.15),
                   test_bgs=test_all[low_idx], test_soft=np.zeros((len(low_idx), 10), np.float32), p_test_bg=1.0)
    Xr, Yr = rv.batch(1600)
    out = dict(Xs=Xs.numpy(), Ys=Ys.numpy(), Xr=Xr.numpy(), Yr=Yr.numpy())
    np.savez(f, **out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", default="sed")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--nch", type=int, default=0, help="feature channels (default 2 for sed, 3 otherwise)")
    ap.add_argument("--mels", type=int, default=64)
    ap.add_argument("--nfft", type=int, default=512)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--per_epoch", type=int, default=1600)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="train on all train clips (no hold-out)")
    ap.add_argument("--no_distract", action="store_true")
    ap.add_argument("--no_specaug", action="store_true")
    ap.add_argument("--pitch", type=int, default=2, help="max pitch shift in 64-mel bins")
    ap.add_argument("--snr_lo", type=float, default=-10)
    ap.add_argument("--snr_hi", type=float, default=12)
    ap.add_argument("--eq", type=float, default=0.0)
    ap.add_argument("--p_gate", type=float, default=0.0)
    ap.add_argument("--n_distract", type=float, default=1.5)
    ap.add_argument("--pseudo", default="", help="npy/npz with 800x10 soft test labels -> real test clips as backgrounds")
    ap.add_argument("--p_test_bg", type=float, default=0.5)
    ap.add_argument("--pseudo_thr", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tta", type=int, default=4)
    a = ap.parse_args()
    nch = a.nch or (2 if a.arch == "sed" else 3)

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    d = load_cache(a.mels, a.nfft)
    tr, _, _ = read_csvs()
    y = train_targets(tr, 1.0)
    dn = denoise(d["train"])
    test_all = np.concatenate([d["public"], d["private"]])
    v1 = np.load(CACHE / "v1_probs.npy")  # v1 ensemble test predictions: only used to pick low-activity RV backgrounds
    low_idx = np.argsort(v1.sum(1))[:200]
    V = make_vals(d, tr, a.mels, a.nfft, low_idx)
    vfl = stationary_floor(test_all)

    tr_idx, _ = holdout_split(tr)
    if a.full:
        tr_idx = np.arange(len(tr))
    kw = dict(birds=dn[tr_idx], labels=y[tr_idx], floors=vfl[0::2] if not a.full else vfl,
              snr_db=(a.snr_lo, a.snr_hi), n_distract=0.0 if a.no_distract else a.n_distract,
              pitch=round(a.pitch * a.mels / 64), eq=a.eq, p_gate=a.p_gate)
    if a.pseudo:
        ps = np.load(a.pseudo)
        ps = ps["soft"] if hasattr(ps, "files") else ps
        kw.update(test_bgs=test_all, test_soft=np.where(ps > a.pseudo_thr, ps, 0.0).astype(np.float32), p_test_bg=a.p_test_bg)
    loader = DataLoader(MixStream(a.mels, a.seed + 1, kw), batch_size=a.bs, num_workers=a.workers,
                        persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)
    it = iter(loader)

    cfg = dict(arch=a.arch, width=a.width, nch=nch, mels=a.mels, nfft=a.nfft, drop=a.drop, pretrained=not a.no_pretrained)
    model = build_model(cfg).to(DEVICE)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (a.per_epoch // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    bce = nn.BCEWithLogitsLoss()
    Xs, Xr = torch.from_numpy(V["Xs"]), torch.from_numpy(V["Xr"])

    log, step = [], 0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for _ in range(a.per_epoch // a.bs):
            xb, yb = next(it)
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            feats = to_features(xb, nch)
            if not a.no_specaug:
                feats = spec_augment(feats, rng, a.mels)
            loss = bce(model(feats), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            dec = min(a.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
                    if pe.dtype.is_floating_point:
                        pe.mul_(dec).add_(pm.detach(), alpha=1 - dec)
                    else:
                        pe.copy_(pm)
            tot += loss.item()
        rec = dict(ep=ep, loss=round(tot / (a.per_epoch // a.bs), 4), sec=round(time.time() - t0, 1))
        if ep % 5 == 0 or ep == a.epochs or ep <= 2:
            rec["sv"] = round(macro_auc(V["Ys"], predict(ema, Xs, nch)), 5)
            rec["rv"] = round(macro_auc(V["Yr"], predict(ema, Xr, nch)), 5)
        log.append(rec)
        print(json.dumps(rec), flush=True)

    torch.save({k: v.cpu() for k, v in ema.state_dict().items()}, CACHE / f"{a.name}.pt")
    ps_ = predict(ema, Xs, nch, tta=a.tta)
    pr_ = predict(ema, Xr, nch, tta=a.tta)
    res = dict(cfg=cfg, args=vars(a), log=log, sv_tta=macro_auc(V["Ys"], ps_), rv_tta=macro_auc(V["Yr"], pr_))
    np.savez(CACHE / f"{a.name}_val.npz", sv=ps_, rv=pr_)
    test = predict(ema, torch.from_numpy(test_all), nch, tta=a.tta)
    np.save(CACHE / f"{a.name}_test.npy", test)
    (CACHE / f"{a.name}.json").write_text(json.dumps(res))
    print("FINAL", a.name, "sv_tta", round(res["sv_tta"], 5), "rv_tta", round(res["rv_tta"], 5), flush=True)


if __name__ == "__main__":
    main()
