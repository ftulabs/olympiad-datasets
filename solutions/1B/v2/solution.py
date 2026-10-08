"""1B E-wallet fraud, v2: logistic spline-GAM with generator-matched interaction terms (PyTorch).

Pipeline
  Data      row-wise features: v1 features + interactions found by residual screening
            (night x new_device, new_device x ratio, far x channel, round x non-p2p, ...) and
            card-testing terms (tiny amount x |log ratio|).
  Model     logistic GAM in PyTorch: piecewise-linear splines (8 quantile knots) on the
            continuous features + linear flags/interactions + free category x channel bias.
            Fitted full-batch with L-BFGS (convex => deterministic, no seeds needed).
  Training  3 x repeated stratified 5-fold CV -> OOF probabilities -> F1-optimal threshold
            (smoothed curve). Final model refit on all labelled rows; test = 0.5 * fold-model
            average + 0.5 * full refit.

Writes public_submission.csv / private_submission.csv next to this file.
Runtime ~2-3 min on 2 CPU threads, < 1 GB RAM.
"""
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA = Path(os.environ.get("DATA1B", "/home/minh/Desktop/olympiad_ai/warmup/1B_fraud_classification/dataset"))
OUT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
TARGET = "is_fraud"
CFG = dict(knots=8, l2_spline=3e-3, l2_bias=1e-6, lbfgs_iters=300, folds=5, repeats=3)

CATS = ["e_commerce", "electronics", "food_delivery", "game_topup",
        "gift_card", "p2p_transfer", "travel", "utilities"]
CHANNELS = ["app", "qr", "web"]
CONT = ["log_amt", "log_avg", "log_ratio", "log_age", "log_dist", "tx", "hour_sin", "hour_cos"]


# ============================================================== Data
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features only (nothing fitted on data -> no leakage between folds/splits)."""
    a, avg = df.amount_vnd, df.avg_amount_30d
    lr = np.log(a / avg)
    p2p = (df.merchant_category == "p2p_transfer") * 1.0
    night = (df.hour <= 5) * 1.0
    newdev = df.new_device * 1.0
    far = (df.distance_from_home_km > 100) * 1.0
    tiny = (a < 20000) * 1.0                       # card-testing amounts are U(1k, 20k)
    r50 = (a % 50000 == 0) * 1.0                   # only multiples of 50k carry signal
    f = pd.DataFrame(index=df.index)
    # --- main effects (continuous -> splines in the model)
    f["log_amt"], f["log_avg"], f["log_ratio"] = np.log(a), np.log(avg), lr
    f["log_age"] = np.log1p(df.account_age_days.clip(upper=3650))           # cap 190-year ages
    f["log_dist"] = np.log1p(df.distance_from_home_km.clip(upper=1000))
    f["tx"] = df.tx_count_24h.clip(upper=12) * 1.0
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * df.hour / 24), np.cos(2 * np.pi * df.hour / 24)
    # --- step / flag effects at the observed breakpoints
    f["ratio_pos"], f["ratio_neg"] = lr.clip(lower=0), (-lr).clip(lower=0)
    f["tiny_amt"], f["round_50k"], f["round_100k"] = tiny, r50, (a % 100000 == 0) * 1.0
    f["night"], f["is_weekend"] = night, df.is_weekend * 1.0
    f["young"] = (df.account_age_days < 30) * 1.0                            # step between 29 and 30 days
    f["tx_hi"], f["tx_x"] = (df.tx_count_24h >= 5) * 1.0, (df.tx_count_24h - 4).clip(lower=0) * 1.0
    f["far"], f["new_device"] = far, newdev
    f["failed_pin"], f["pin2"] = df.failed_pin_24h.clip(upper=4) * 1.0, (df.failed_pin_24h >= 2) * 1.0
    # --- interactions (found by score-test screening of additive-model residuals)
    f["r50_np2p"] = r50 * (1 - p2p)                # round amounts are normal for p2p transfers
    f["night_newdev"], f["night_p2p"] = night * newdev, night * p2p
    f["newdev_rpos"], f["newdev_rneg"] = newdev * f.ratio_pos, newdev * f.ratio_neg
    f["far_qr"], f["far_web"] = far * (df.channel == "qr"), far * (df.channel == "web")
    f["tiny_rneg"] = tiny * f.ratio_neg            # card testing: tiny AND far below the user's usual spend
    f["lr_neg2"] = (lr < -2.5) * 1.0
    f["night_r50"] = night * r50
    f["r50_big"] = r50 * (1 - p2p) * (a >= 1e6)
    f["cc"] = (df.merchant_category.map({c: i for i, c in enumerate(CATS)}).fillna(0).astype(int) * 3
               + df.channel.map({c: i for i, c in enumerate(CHANNELS)}).fillna(0).astype(int))
    return f


# ============================================================== Model
class SplineGAM(nn.Module):
    """logit = w . [x, relu(x_c - knot)] + b[category x channel]."""

    def __init__(self, n_in: int, n_basis: int) -> None:
        super().__init__()
        self.lin = nn.Linear(n_in + n_basis, 1)
        self.cc_bias = nn.Embedding(24, 1)
        nn.init.zeros_(self.cc_bias.weight)
        self.n_in = n_in

    def forward(self, xb: torch.Tensor, cc: torch.Tensor) -> torch.Tensor:
        return self.lin(xb).squeeze(-1) + self.cc_bias(cc).squeeze(-1)

    def penalty(self, l2_spline: float, l2_bias: float) -> torch.Tensor:
        return l2_spline * (self.lin.weight[:, self.n_in:] ** 2).sum() + l2_bias * (self.cc_bias.weight ** 2).mean()


class Design:
    """Standardisation + spline knots fitted on the training rows only."""

    def __init__(self, X: pd.DataFrame, knots: int) -> None:
        self.cols = [c for c in X.columns if c != "cc"]
        self.mu, self.sd = X[self.cols].mean(), X[self.cols].std() + 1e-6
        z = ((X[self.cols] - self.mu) / self.sd)
        self.ci = [self.cols.index(c) for c in CONT]
        qs = np.linspace(0.05, 0.95, knots)
        self.knots = torch.tensor(np.stack([np.quantile(z.iloc[:, j], qs) for j in self.ci]), dtype=torch.float32)

    def __call__(self, X: pd.DataFrame):
        x = torch.tensor(((X[self.cols] - self.mu) / self.sd).values, dtype=torch.float32)
        xc = x[:, self.ci]
        hinges = [F.relu(xc - self.knots[:, j]) for j in range(self.knots.shape[1])]
        return torch.cat([x] + hinges, 1).to(DEVICE), torch.tensor(X.cc.values).to(DEVICE)


# ============================================================== Training & inference
def fit_predict(Xa: pd.DataFrame, ya: np.ndarray, X_eval: list) -> list:
    design = Design(Xa, CFG["knots"])
    xb, cc = design(Xa)
    yt = torch.tensor(ya, dtype=torch.float32, device=DEVICE)
    model = SplineGAM(len(design.cols), xb.shape[1] - len(design.cols)).to(DEVICE)
    opt = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=CFG["lbfgs_iters"], history_size=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(xb, cc), yt) + model.penalty(CFG["l2_spline"], CFG["l2_bias"])
        loss.backward()
        return loss

    opt.step(closure)
    model.eval()
    with torch.no_grad():
        return [torch.sigmoid(model(*design(d))).cpu().numpy() for d in X_eval]


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

    R, K = CFG["repeats"], CFG["folds"]
    oof = np.zeros(len(y))
    p_pub, p_pri = np.zeros(len(Xpu)), np.zeros(len(Xpr))
    for r in range(R):
        oof_r = np.zeros(len(y))
        for tr_idx, va_idx in StratifiedKFold(K, shuffle=True, random_state=r).split(X, y):
            pv, pu, pr = fit_predict(X.iloc[tr_idx], y[tr_idx], [X.iloc[va_idx], Xpu, Xpr])
            oof_r[va_idx] = pv
            p_pub += pu / (R * K)
            p_pri += pr / (R * K)
        print(f"repeat {r}: OOF AP={average_precision_score(y, oof_r):.4f} "
              f"LL={log_loss(y, oof_r):.5f} F1={best_threshold(y, oof_r)[1]:.4f}", flush=True)
        oof += oof_r / R

    thr, f1 = best_threshold(y, oof)
    print(f"OOF (avg of {R} repeats) AP={average_precision_score(y, oof):.4f} F1={f1:.4f} threshold={thr:.3f}")

    full_pub, full_pri = fit_predict(X, y, [Xpu, Xpr])          # final refit on all labelled rows
    p_pub, p_pri = 0.5 * p_pub + 0.5 * full_pub, 0.5 * p_pri + 0.5 * full_pri
    for name, ids, p in [("public", public.id, p_pub), ("private", private.id, p_pri)]:
        pred = (p >= thr).astype(int)
        pd.DataFrame({"id": ids, TARGET: pred}).to_csv(OUT / f"{name}_submission.csv", index=False)
        print(f"{name}: {len(pred)} rows, predicted fraud rate {pred.mean():.3%}")
    np.savez_compressed(OUT / "oof_and_test_probs.npz", oof=oof, pub=p_pub, pri=p_pri, thr=thr)


if __name__ == "__main__":
    main()
