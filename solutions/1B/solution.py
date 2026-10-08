"""1B E-wallet fraud: end-to-end pipeline (feature engineering + residual MLP in PyTorch).

Writes public_submission.csv and private_submission.csv next to this file.
CPU only, single thread, ~3-4 minutes.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold

torch.set_num_threads(1)

DATA = Path("/home/minh/Desktop/olympiad_ai/warmup/1B_fraud_classification/dataset")
OUT = Path(__file__).resolve().parent
TARGET = "is_fraud"
CFG = dict(hidden=64, depth=1, drop=0.2, lr=2e-3, wd=1e-4, epochs=8, bs=512, folds=5, seeds=3)

CATS = ["e_commerce", "electronics", "food_delivery", "game_topup",
        "gift_card", "p2p_transfer", "travel", "utilities"]
CHANNELS = ["app", "qr", "web"]


# ----------------------------------------------------------------- Data
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering (no statistics learned from data -> no leakage)."""
    a, avg = df.amount_vnd, df.avg_amount_30d
    f = pd.DataFrame(index=df.index)
    f["log_amt"] = np.log(a)
    f["log_avg"] = np.log(avg)
    f["log_ratio"] = np.log(a / avg)                    # amount vs the user's own 30d average
    f["ratio_pos"] = f.log_ratio.clip(lower=0)          # U-shaped risk: unusually large ...
    f["ratio_neg"] = (-f.log_ratio).clip(lower=0)       # ... or unusually small (card testing)
    f["tiny_amt"] = (a < 20000).astype(float)           # micro-payments: 27% fraud vs 4.7%
    f["round_50k"] = (a % 50000 == 0).astype(float)     # round amounts: ~2.5x fraud rate
    f["round_10k"] = (a % 10000 == 0).astype(float)
    f["hour_sin"] = np.sin(2 * np.pi * df.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * df.hour / 24)
    f["night"] = (df.hour <= 5).astype(float)
    f["is_weekend"] = df.is_weekend.astype(float)
    f["log_age"] = np.log1p(df.account_age_days.clip(upper=3650))  # cap impossible ages (>10y)
    f["young"] = (df.account_age_days <= 30).astype(float)          # step: 15% vs 4% fraud
    f["tx"] = df.tx_count_24h.clip(upper=12).astype(float)
    f["tx_hi"] = (df.tx_count_24h >= 5).astype(float)
    f["log_dist"] = np.log1p(df.distance_from_home_km.clip(upper=1000))
    f["far"] = (df.distance_from_home_km > 100).astype(float)
    f["new_device"] = df.new_device.astype(float)
    f["failed_pin"] = df.failed_pin_24h.clip(upper=4).astype(float)
    f["cat"] = df.merchant_category.map({c: i for i, c in enumerate(CATS)}).fillna(0).astype(int)
    f["ch"] = df.channel.map({c: i for i, c in enumerate(CHANNELS)}).fillna(0).astype(int)
    return f


# ----------------------------------------------------------------- Model
class ResBlock(nn.Module):
    def __init__(self, h: int, drop: float) -> None:
        super().__init__()
        self.bn, self.l1, self.l2, self.drop = nn.BatchNorm1d(h), nn.Linear(h, h), nn.Linear(h, h), nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.l2(self.drop(F.silu(self.l1(F.silu(self.bn(x))))))


class FraudNet(nn.Module):
    """Numeric features + embeddings for category, channel and category x channel."""

    def __init__(self, n_num: int, hidden: int, depth: int, drop: float) -> None:
        super().__init__()
        self.e_cat, self.e_ch, self.e_x = nn.Embedding(8, 4), nn.Embedding(3, 2), nn.Embedding(24, 4)
        self.inp = nn.Linear(n_num + 10, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden, drop) for _ in range(depth)])
        self.head = nn.Sequential(nn.BatchNorm1d(hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, xn: torch.Tensor, cat: torch.Tensor, ch: torch.Tensor) -> torch.Tensor:
        z = torch.cat([xn, self.e_cat(cat), self.e_ch(ch), self.e_x(cat * 3 + ch)], dim=1)
        return self.head(self.blocks(self.inp(z))).squeeze(-1)


# ----------------------------------------------------------------- Training & inference
def fit_predict(Xa: pd.DataFrame, ya: np.ndarray, X_eval: list, seed: int) -> list:
    torch.manual_seed(seed)
    np.random.seed(seed)
    num = [c for c in Xa.columns if c not in ("cat", "ch")]
    mu, sd = Xa[num].mean(), Xa[num].std() + 1e-6       # scaler fitted on the training fold only

    def to_t(d: pd.DataFrame):
        return (torch.tensor(((d[num] - mu) / sd).values, dtype=torch.float32),
                torch.tensor(d.cat.values), torch.tensor(d.ch.values))

    xa, ca, ha = to_t(Xa)
    yt = torch.tensor(ya, dtype=torch.float32)
    model = FraudNet(len(num), CFG["hidden"], CFG["depth"], CFG["drop"])
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    n, bs = len(ya), CFG["bs"]
    steps = (n + bs - 1) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["lr"], total_steps=steps * CFG["epochs"], pct_start=0.2)
    for _ in range(CFG["epochs"]):
        model.train()
        perm = torch.randperm(n)
        for i in range(steps):
            idx = perm[i * bs:(i + 1) * bs]
            if len(idx) < 2:
                sched.step()
                continue
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xa[idx], ca[idx], ha[idx]), yt[idx])
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    with torch.no_grad():
        return [torch.sigmoid(model(*to_t(d))).numpy() for d in X_eval]


def best_threshold(y: np.ndarray, p: np.ndarray) -> tuple:
    ts = np.quantile(p, np.linspace(0.85, 0.995, 300))
    f1s = np.array([f1_score(y, p >= t) for t in ts])
    f1s_smooth = np.convolve(f1s, np.ones(9) / 9, mode="same")  # robust to a lucky single cut
    i = int(np.argmax(f1s_smooth))
    return float(ts[i]), float(f1s[i])


def main() -> None:
    train = pd.read_csv(DATA / "train/train.csv")
    public = pd.read_csv(DATA / "public_test/public_test.csv")
    private = pd.read_csv(DATA / "private_test/private_test.csv")
    X, Xpu, Xpr = make_features(train), make_features(public), make_features(private)
    y = train[TARGET].values

    oof = np.zeros(len(y))
    p_pub, p_pri = np.zeros(len(Xpu)), np.zeros(len(Xpr))
    n_models = CFG["folds"] * CFG["seeds"]
    skf = StratifiedKFold(CFG["folds"], shuffle=True, random_state=0)
    for k, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        for s in range(CFG["seeds"]):
            pv, pu, pr = fit_predict(X.iloc[tr_idx], y[tr_idx], [X.iloc[va_idx], Xpu, Xpr], seed=100 * k + s)
            oof[va_idx] += pv / CFG["seeds"]
            p_pub += pu / n_models
            p_pri += pr / n_models
        print(f"fold {k}: AP={average_precision_score(y[va_idx], oof[va_idx]):.4f}", flush=True)

    thr, f1 = best_threshold(y, oof)
    print(f"OOF AP={average_precision_score(y, oof):.4f}  OOF F1={f1:.4f} at threshold {thr:.3f}")
    for name, ids, p in [("public", public.id, p_pub), ("private", private.id, p_pri)]:
        pred = (p >= thr).astype(int)
        pd.DataFrame({"id": ids, TARGET: pred}).to_csv(OUT / f"{name}_submission.csv", index=False)
        print(f"{name}: {len(pred)} rows, predicted fraud rate {pred.mean():.3%}")


if __name__ == "__main__":
    main()
