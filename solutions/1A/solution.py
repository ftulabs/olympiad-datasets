"""1A - VN30 next-day return: final end-to-end pipeline.

Data   : feature engineering (log volume ratio, vol-normalised returns,
         volume-regime interactions, NaN flag), quantile clipping, standard scaling.
Model  : PyTorch residual MLP + ticker/sector/day-of-week embeddings + wide
         (linear) path + per-ticker bias.
Train  : Huber loss, AdamW, OneCycle LR, fixed epoch count chosen on a
         time-based validation (2023, 2024 folds), retrain on all of 2021-2024,
         ensemble of seeds.

Usage: python solution.py   (CPU only, writes the two CSVs next to this file)
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(1)

DATA_DIR = Path("/home/minh/Desktop/olympiad_ai/warmup/1A_stock_regression/dataset")
OUT_DIR = Path(__file__).resolve().parent
TARGET = "next_ret"

# ---- hyperparameters (selected on own time-based CV, see README) ----
CFG = dict(
    hidden=128,
    depth=2,
    dropout=0.4,
    emb=(8, 4, 3),  # ticker, sector, day_of_week
    epochs=40,
    batch_size=512,
    lr=3e-3,
    weight_decay=1e-4,
    huber_delta=1.0,
    seeds=tuple(range(8)),
)

RAW = ["ret_1d", "ret_5d", "ret_20d", "volatility_20d", "rsi_14",
       "volume_ratio", "foreign_net_buy_bn", "market_ret_1d"]  # usd_vnd_change dropped


# ============================== Data ==============================
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric features. Everything is row-local, so no look-ahead leakage."""
    x = df[RAW].copy()
    lvr = np.log(df["volume_ratio"])
    vol = df["volatility_20d"]
    x["foreign_net_buy_bn"] = df["foreign_net_buy_bn"].fillna(0.0)
    x["fnan"] = df["foreign_net_buy_bn"].isna().astype(float)
    x["volume_ratio"] = lvr                                  # heavy right tail -> log
    x["r1v"] = df["ret_1d"] / vol                            # vol-normalised returns
    x["r5v"] = df["ret_5d"] / vol
    x["r20v"] = df["ret_20d"] / vol
    x["r1_lvr"] = df["ret_1d"] * lvr                         # reversal (low vol.) vs momentum (high vol.)
    x["mkt_lvr"] = df["market_ret_1d"] * lvr
    x["hivol"] = (df["volume_ratio"] > 1.3).astype(float)
    return x


class Scaler:
    """Clip at train 0.5/99.5% quantiles, then standardise (fit on train only)."""

    def fit(self, x: pd.DataFrame) -> "Scaler":
        self.lo, self.hi = x.quantile(0.005), x.quantile(0.995)
        xc = x.clip(self.lo, self.hi, axis=1)
        self.mu, self.sd = xc.mean(), xc.std() + 1e-6
        return self

    def transform(self, x: pd.DataFrame) -> torch.Tensor:
        z = (x.clip(self.lo, self.hi, axis=1) - self.mu) / self.sd
        return torch.tensor(z.values, dtype=torch.float32)


class CatEncoder:
    def fit(self, df: pd.DataFrame) -> "CatEncoder":
        self.tick = {t: i for i, t in enumerate(sorted(df["ticker"].unique()))}
        self.sec = {s: i for i, s in enumerate(sorted(df["sector"].unique()))}
        return self

    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        t = df["ticker"].map(self.tick)
        s = df["sector"].map(self.sec)
        if t.isna().any() or s.isna().any():
            raise ValueError("unseen ticker/sector in test data")
        arr = np.stack([t.values, s.values, df["day_of_week"].values], 1).astype(np.int64)
        return torch.tensor(arr)


# ============================== Model ==============================
class ResBlock(nn.Module):
    def __init__(self, h: int, p: float) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(h), nn.SiLU(), nn.Dropout(p), nn.Linear(h, h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.f(x)


class Net(nn.Module):
    """Deep path (residual MLP) + wide linear path + per-ticker bias."""

    def __init__(self, n_num: int, n_tick: int, n_sec: int, hidden: int, depth: int,
                 dropout: float, emb: tuple[int, int, int]) -> None:
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, d) for n, d in zip((n_tick, n_sec, 5), emb)])
        d_in = n_num + sum(emb)
        self.inp = nn.Linear(d_in, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(depth - 1)])
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 1))
        self.wide = nn.Linear(d_in, 1)
        self.tick_bias = nn.Embedding(n_tick, 1)
        nn.init.zeros_(self.tick_bias.weight)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x] + [e(c[:, i]) for i, e in enumerate(self.embs)], 1)
        h = self.inp(z)
        for b in self.blocks:
            h = b(h)
        return (self.out(h) + self.wide(z) + self.tick_bias(c[:, :1]).squeeze(1)).squeeze(-1)


# ======================= Training & inference =======================
def train_one(x: torch.Tensor, c: torch.Tensor, y: torch.Tensor, n_tick: int, n_sec: int,
              seed: int) -> Net:
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = Net(x.shape[1], n_tick, n_sec, CFG["hidden"], CFG["depth"], CFG["dropout"], CFG["emb"])
    opt = torch.optim.AdamW(m.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    n, bs = len(y), CFG["batch_size"]
    steps = CFG["epochs"] * ((n + bs - 1) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["lr"], total_steps=steps, pct_start=0.2)
    crit = nn.HuberLoss(delta=CFG["huber_delta"])
    for _ in range(CFG["epochs"]):
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            crit(m(x[idx], c[idx]), y[idx]).backward()
            opt.step()
            sch.step()
    return m


@torch.no_grad()
def predict(m: Net, x: torch.Tensor, c: torch.Tensor) -> np.ndarray:
    m.eval()
    return m(x, c).numpy()


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train/train.csv")
    public = pd.read_csv(DATA_DIR / "public_test/public_test.csv")
    private = pd.read_csv(DATA_DIR / "private_test/private_test.csv")

    f_train = make_features(train)
    scaler = Scaler().fit(f_train)
    enc = CatEncoder().fit(train)
    x_tr, c_tr = scaler.transform(f_train), enc.transform(train)
    y_tr = torch.tensor(train[TARGET].values, dtype=torch.float32)
    tests = {name: (scaler.transform(make_features(df)), enc.transform(df), df)
             for name, df in (("public", public), ("private", private))}

    preds = {name: [] for name in tests}
    for seed in CFG["seeds"]:
        m = train_one(x_tr, c_tr, y_tr, len(enc.tick), len(enc.sec), seed)
        for name, (x, c, _) in tests.items():
            preds[name].append(predict(m, x, c))
        print(f"seed {seed} done", flush=True)

    for name, (_, _, df) in tests.items():
        out = pd.DataFrame({"id": df["id"], TARGET: np.mean(preds[name], 0)})
        assert out[TARGET].notna().all() and len(out) == len(df)
        out.to_csv(OUT_DIR / f"{name}_submission.csv", index=False)
        print(f"wrote {name}_submission.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
