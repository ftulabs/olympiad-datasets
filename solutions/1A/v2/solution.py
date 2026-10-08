"""1A - VN30 next-day return, v2: structural (data-generating-process) model.

Data   : EDA recovered the hidden generator. The target is a sum of a few simple terms:
         ticker alpha, sector beta x market return, tanh(RSI), tanh(foreign flow),
         ret_20d / sqrt(vol), a Monday effect, and a ret_1d/vol term that switches
         regime at volume_ratio 1.2 and 2.0 (reversal, then momentum, then a breakout
         step). Its slope is scaled per ticker. The noise is Student-t (nu about 4)
         with scale proportional to sqrt(vol).
Model  : (a) a PyTorch parametric model of exactly that form (about 90 parameters),
         fitted by maximum likelihood with a heteroscedastic Student-t loss and LBFGS.
         Three variants differ only in the breakout term and are averaged.
         (b) The v1 wide-and-deep MLP, used as a diverse member with a 10% weight.
Train  : refit on all 2021-2024 data. The blend weight and every structural choice
         came from own time-based CV (2023 and 2024 forward folds plus a
         leave-one-year-out check), see README.

Usage: python solution.py          -> writes public/private_submission.csv next to this file
       python solution.py --cv     -> reproduces the forward-fold CV of the final blend
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # used for the NN member

DATA_DIR = Path(os.environ.get(
    "DATA_DIR", "/home/minh/Desktop/olympiad_ai/warmup/1A_stock_regression/dataset"))
OUT_DIR = Path(__file__).resolve().parent
TARGET = "next_ret"
THR_MID, THR_TOP = 1.2, 2.0          # volume_ratio regime thresholds (found in EDA, CV-confirmed)
PARAM_VARIANTS = ({"top": "free"}, {"top": "shared"}, {"top": "shared_nolin"})
NN_SEEDS = tuple(range(8))
W_NN = 0.10                           # blend weight of the NN member (CV-selected)


# ================================ Data ================================
def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(DATA_DIR / "train/train.csv")
    pu = pd.read_csv(DATA_DIR / "public_test/public_test.csv")
    pr = pd.read_csv(DATA_DIR / "private_test/private_test.csv")
    for d in (tr, pu, pr):
        d["year"] = d["date"].str[:4].astype(int)
    return tr, pu, pr


class Vocab:
    def __init__(self, train: pd.DataFrame) -> None:
        self.tk = {t: i for i, t in enumerate(sorted(train["ticker"].unique()))}
        self.sc = {s: i for i, s in enumerate(sorted(train["sector"].unique()))}

    def codes(self, d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        t, s = d["ticker"].map(self.tk), d["sector"].map(self.sc)
        if t.isna().any() or s.isna().any():
            raise ValueError("unseen ticker/sector")
        return t.values.astype(np.int64), s.values.astype(np.int64)


def to_tensors(d: pd.DataFrame, voc: Vocab) -> dict[str, torch.Tensor]:
    f = lambda c: torch.tensor(np.asarray(c, dtype=np.float64))
    tk, sc = voc.codes(d)
    out = dict(tk=torch.tensor(tk), sc=torch.tensor(sc), mon=f(d["day_of_week"] == 0),
               rsi=f(d["rsi_14"]), fnb=f(d["foreign_net_buy_bn"].fillna(0.0)),
               fnan=f(d["foreign_net_buy_bn"].isna()), r20=f(d["ret_20d"]), r1=f(d["ret_1d"]),
               vol=f(d["volatility_20d"]), vr=f(d["volume_ratio"]), mkt=f(d["market_ret_1d"]))
    if TARGET in d:
        out["y"] = f(d[TARGET])
    return out


# ============================ Model (a): structural ============================
class StructuralModel(nn.Module):
    """y = alpha_ticker + beta_sector*mkt + a*tanh((rsi-50)/s_r) + b*tanh(fnb/s_f)
           + c*ret_20d/vol^k20 + monday + g_ticker*[lo: c_lo*x + mid/top: c_mid*x] + top: breakout
       with x = ret_1d/vol^k1; noise ~ Student-t(nu), scale = exp(l0 + l1*log vol)."""

    def __init__(self, n_tk: int, n_sc: int, top: str) -> None:
        super().__init__()
        self.top = top
        p = lambda v: nn.Parameter(torch.tensor(v, dtype=torch.float64))
        self.alpha, self.beta, self.g = p([0.0] * n_tk), p([0.0] * n_sc), p([1.0] * n_tk)
        self.mon, self.cfn, self.c0 = p(-0.26), p(0.0), p([0.0, 0.0])
        self.a_rsi, self.ls_rsi = p(-0.9), p(math.log(12.0))
        self.a_f, self.ls_f = p(0.55), p(math.log(45.0))
        self.c20, self.k20, self.k1 = p(0.07), p(0.5), p(1.0)
        self.c_lo, self.c_mid = p(-0.43), p(0.2)
        self.c_top, self.b_top, self.ls_top = p(0.45), p(0.1), p(math.log(0.1))
        self.lsig, self.lnu = p([-0.5, 0.45]), p(0.75)

    def forward(self, d: dict[str, torch.Tensor]) -> torch.Tensor:
        v, vr = d["vol"], d["vr"]
        lo = (vr < THR_MID).double()
        top = (vr >= THR_TOP).double()
        mid = 1.0 - lo - top
        x = d["r1"] / v ** self.k1
        out = (self.alpha[d["tk"]] + self.beta[d["sc"]] * d["mkt"]
               + self.a_rsi * torch.tanh((d["rsi"] - 50.0) / self.ls_rsi.exp())
               + self.a_f * torch.tanh(d["fnb"] / self.ls_f.exp())
               + self.c20 * d["r20"] / v ** self.k20
               + self.mon * d["mon"] + self.cfn * d["fnan"] + self.c0[0] * mid + self.c0[1] * top)
        g = self.g[d["tk"]]
        step = self.c_top * torch.tanh(x / self.ls_top.exp())
        if self.top == "free":          # breakout regime has its own (unscaled) slope
            r1 = g * (lo * self.c_lo * x + mid * self.c_mid * x) + top * (step + self.b_top * x)
        else:                           # mid slope (ticker-scaled) continues into top regime
            b = self.b_top if self.top == "shared" else 0.0
            r1 = g * (lo * self.c_lo * x + (mid + top) * self.c_mid * x) + top * (step + b * x)
        return out + r1

    def nll(self, d: dict[str, torch.Tensor]) -> torch.Tensor:
        r = d["y"] - self(d)
        sig = torch.exp(self.lsig[0] + self.lsig[1] * torch.log(d["vol"]))
        nu = self.lnu.exp() + 2.0
        return (0.5 * (nu + 1) * torch.log1p((r / sig) ** 2 / nu) + torch.log(sig)
                + torch.lgamma(nu / 2) - torch.lgamma((nu + 1) / 2) + 0.5 * torch.log(nu)).mean()


def fit_structural(d: dict[str, torch.Tensor], n_tk: int, n_sc: int, top: str) -> StructuralModel:
    m = StructuralModel(n_tk, n_sc, top)
    opt = torch.optim.LBFGS(m.parameters(), lr=1, max_iter=1000, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10, tolerance_change=1e-14)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = m.nll(d)
        loss.backward()
        return loss

    opt.step(closure)
    return m


def structural_predict(train: pd.DataFrame, tests: list[pd.DataFrame]) -> list[np.ndarray]:
    voc = Vocab(train)
    dtr = to_tensors(train, voc)
    outs = [np.zeros(len(t)) for t in tests]
    for var in PARAM_VARIANTS:
        m = fit_structural(dtr, len(voc.tk), len(voc.sc), var["top"])
        with torch.no_grad():
            for i, t in enumerate(tests):
                outs[i] += m(to_tensors(t, voc)).numpy() / len(PARAM_VARIANTS)
    return outs


# ======================= Model (b): wide-and-deep MLP (v1) =======================
NN_RAW = ["ret_1d", "ret_5d", "ret_20d", "volatility_20d", "rsi_14",
          "volume_ratio", "foreign_net_buy_bn", "market_ret_1d"]


def nn_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df[NN_RAW].copy()
    lvr, vol = np.log(df["volume_ratio"]), df["volatility_20d"]
    x["foreign_net_buy_bn"] = df["foreign_net_buy_bn"].fillna(0.0)
    x["fnan"] = df["foreign_net_buy_bn"].isna().astype(float)
    x["volume_ratio"] = lvr
    x["r1v"], x["r5v"], x["r20v"] = df["ret_1d"] / vol, df["ret_5d"] / vol, df["ret_20d"] / vol
    x["r1_lvr"], x["mkt_lvr"] = df["ret_1d"] * lvr, df["market_ret_1d"] * lvr
    x["hivol"] = (df["volume_ratio"] > 1.3).astype(float)
    return x


class Scaler:
    def fit(self, x: pd.DataFrame) -> "Scaler":
        self.lo, self.hi = x.quantile(0.005), x.quantile(0.995)
        xc = x.clip(self.lo, self.hi, axis=1)
        self.mu, self.sd = xc.mean(), xc.std() + 1e-6
        return self

    def transform(self, x: pd.DataFrame) -> torch.Tensor:
        z = (x.clip(self.lo, self.hi, axis=1) - self.mu) / self.sd
        return torch.tensor(z.values, dtype=torch.float32)


class ResBlock(nn.Module):
    def __init__(self, h: int, p: float) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(h), nn.SiLU(), nn.Dropout(p), nn.Linear(h, h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.f(x)


class WideDeep(nn.Module):
    def __init__(self, n_num: int, n_tk: int, n_sc: int, hidden: int = 128, dropout: float = 0.4,
                 emb: tuple[int, int, int] = (8, 4, 3)) -> None:
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, k) for n, k in zip((n_tk, n_sc, 5), emb)])
        d_in = n_num + sum(emb)
        self.inp, self.block = nn.Linear(d_in, hidden), ResBlock(hidden, dropout)
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 1))
        self.wide, self.tick_bias = nn.Linear(d_in, 1), nn.Embedding(n_tk, 1)
        nn.init.zeros_(self.tick_bias.weight)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x] + [e(c[:, i]) for i, e in enumerate(self.embs)], 1)
        h = self.block(self.inp(z))
        return (self.out(h) + self.wide(z) + self.tick_bias(c[:, :1]).squeeze(1)).squeeze(-1)


def nn_predict(train: pd.DataFrame, tests: list[pd.DataFrame], seeds=NN_SEEDS) -> list[np.ndarray]:
    voc = Vocab(train)
    cats = lambda d: torch.tensor(np.stack([*voc.codes(d), d["day_of_week"].values], 1).astype(np.int64))
    ftr = nn_features(train)
    sc = Scaler().fit(ftr)
    x, c = sc.transform(ftr).to(DEVICE), cats(train).to(DEVICE)
    y = torch.tensor(train[TARGET].values, dtype=torch.float32, device=DEVICE)
    xt = [(sc.transform(nn_features(t)).to(DEVICE), cats(t).to(DEVICE)) for t in tests]
    outs = [np.zeros(len(t)) for t in tests]
    epochs, bs, lr = 40, 512, 3e-3
    for seed in seeds:
        torch.manual_seed(seed)
        m = WideDeep(x.shape[1], len(voc.tk), len(voc.sc)).to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
        n = len(y)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((n + bs - 1) // bs),
                                                  pct_start=0.2)
        crit = nn.HuberLoss(delta=1.0)
        for _ in range(epochs):
            m.train()
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                crit(m(x[idx], c[idx]), y[idx]).backward()
                opt.step()
                sch.step()
        m.eval()
        with torch.no_grad():
            for i, (xx, cc) in enumerate(xt):
                outs[i] += m(xx, cc).cpu().numpy() / len(seeds)
    return outs


# ============================ Training & inference ============================
def predict_all(train: pd.DataFrame, tests: list[pd.DataFrame], seeds=NN_SEEDS) -> list[np.ndarray]:
    ps = structural_predict(train, tests)
    pn = nn_predict(train, tests, seeds)
    return [(1 - W_NN) * a + W_NN * b for a, b in zip(ps, pn)]


def run_cv() -> None:
    tr, _, _ = load()
    for y in (2023, 2024):
        a, b = tr[tr.year < y], tr[tr.year == y]
        ps, = structural_predict(a, [b])
        pn, = nn_predict(a, [b], seeds=(0, 1, 2))
        r = lambda p: float(np.sqrt(np.mean((b[TARGET].values - p) ** 2)))
        print(f"fold {y}: structural {r(ps):.4f}  nn {r(pn):.4f}  blend {r((1 - W_NN) * ps + W_NN * pn):.4f}")


def main() -> None:
    tr, pu, pr = load()
    p_pub, p_pri = predict_all(tr, [pu, pr])
    for name, df, p in (("public", pu, p_pub), ("private", pr, p_pri)):
        out = pd.DataFrame({"id": df["id"], TARGET: p})
        assert out[TARGET].notna().all() and len(out) == len(df)
        out.to_csv(OUT_DIR / f"{name}_submission.csv", index=False)
        print(f"wrote {name}_submission.csv ({len(out)} rows)")


if __name__ == "__main__":
    run_cv() if "--cv" in sys.argv else main()
