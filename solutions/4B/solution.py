"""4B - Tết demand forecasting. End-to-end solution.

Season-aligned direct multi-horizon framing:
  every training example = (store, item, target day) inside a 64-day window that starts
  right after an "origin" day; all level features come from history <= origin.
  Final forecast: origin 2025-12-31, window 2026-01-01 .. 2026-03-04 (Tết = day 48).

Model: global PyTorch MLP with embeddings (store, item, category, store type, city, weekday,
days-to-Tết, lunar day, holiday, horizon week) + numeric features, multiplicative offset
log(baseline level) and a revenue-weighted Poisson loss (metric = revenue-weighted WAPE).

Usage:
  python solution.py                 # final fit on all history, writes the two CSVs
  python solution.py --backtest 2025 # train on data <= 2024-12-31, score Jan 1 - Mar 4 2025
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/home/minh/Desktop/olympiad_ai/warmup/4B_tet_demand_forecast/dataset"
H = 64  # forecast horizon (Jan 1 .. Mar 4)
TET = {2023: "2023-01-22", 2024: "2024-02-10", 2025: "2025-01-29", 2026: "2026-02-17"}
K_MIN, K_MAX = -63, 45  # days-to-Tết range that gets its own embedding
SEASONAL = ["I15", "I16", "I30", "I48"]  # Tết-only products
CFG = dict(
    base_long=56, base_short=28, young_days=60, young_window=14,
    epochs=12, lr=6e-3, batch=4096, hidden=192, seeds=5, wd=1e-5,
    extra_origins=True, tet_w=1.0, use_ly=False, i48_ratio=None, young_growth=True, calib=1.0,
)


# ----------------------------------------------------------------------------- data
def load() -> dict:
    tr = pd.read_csv(f"{DATA}/train/train.csv", parse_dates=["date"])
    pu = pd.read_csv(f"{DATA}/public_test/public_test.csv", parse_dates=["date"])
    pr = pd.read_csv(f"{DATA}/private_test/private_test.csv", parse_dates=["date"])
    items = pd.read_csv(f"{DATA}/train/items.csv")
    stores = pd.read_csv(f"{DATA}/train/stores.csv")
    cal = pd.read_csv(f"{DATA}/train/calendar.csv", parse_dates=["date"])
    pu["part"], pr["part"], tr["part"] = "public", "private", "train"
    df = pd.concat([tr, pu.drop(columns="weight"), pr.drop(columns="weight")], ignore_index=True)
    return dict(df=df, items=items, stores=stores, cal=cal)


def build_panel(d: dict) -> dict:
    """Dense [series x day] arrays: sales, exists, discount, normal-day mask, stock-out mask."""
    df, items, stores, cal = d["df"], d["items"], d["stores"], d["cal"]
    days = pd.date_range("2022-11-01", "2026-03-04")
    D = len(days)
    ser = df[["store_id", "item_id"]].drop_duplicates().sort_values(["store_id", "item_id"])
    ser = ser.reset_index(drop=True)
    sidx = {(s, i): n for n, (s, i) in enumerate(zip(ser.store_id, ser.item_id))}
    S = len(ser)
    r = pd.MultiIndex.from_frame(df[["store_id", "item_id"]]).map(sidx).to_numpy()
    c = (df["date"] - days[0]).dt.days.to_numpy()
    sales = np.full((S, D), np.nan, np.float32)
    exists = np.zeros((S, D), bool)
    disc = np.zeros((S, D), np.float32)
    rowid = np.full((S, D), -1, np.int64)
    sales[r, c] = df["sales"].to_numpy(np.float32)
    exists[r, c] = True
    disc[r, c] = df["discount_pct"].to_numpy(np.float32)
    rowid[r, c] = df["id"].to_numpy()

    cal = cal.set_index("date").reindex(days)
    # days-to-Tết: each day belongs to the season of the nearest Tết
    tets = np.array([(pd.Timestamp(v) - days[0]).days for v in TET.values()])
    dn = np.arange(D)
    near = tets[np.abs(dn[:, None] - tets[None, :]).argmin(1)]
    k = dn - near
    lunar_new_year = (cal["lunar_month"].to_numpy() == 1) & (cal["lunar_day"].to_numpy() == 1)

    # store closed days (all items 0): Tết day + S05 renovation in July 2024
    tot = np.nansum(np.where(exists, sales, 0), 0)
    closed = np.zeros((S, D), bool)
    for st in ser.store_id.unique():
        m = (ser.store_id == st).to_numpy()
        stot = np.nansum(np.where(exists[m], sales[m], 0), 0)
        has = exists[m].any(0) & (dn <= (pd.Timestamp("2025-12-31") - days[0]).days)
        closed[m] = (has & (stot == 0))[None, :]
    del tot

    # stock-outs: zero on a day where the local median demand is clearly positive
    sdf = pd.DataFrame(np.where(closed, np.nan, sales).T)
    med = sdf.rolling(29, center=True, min_periods=7).median().to_numpy().T
    zero = (sales == 0) & ~closed
    prev0 = np.zeros_like(zero)
    prev0[:, 1:] = zero[:, :-1]
    next0 = np.zeros_like(zero)
    next0[:, :-1] = zero[:, 1:]
    stockout = zero & ((med >= 4) | ((prev0 | next0) & (med >= 2)))

    # post-promo: within 7 days after a promotion ended
    promo = disc > 0
    post = np.zeros((S, D), np.float32)
    last_end = np.full(S, -999)
    for t in range(1, D):
        ended = promo[:, t - 1] & ~promo[:, t]
        last_end = np.where(ended, t, last_end)
        gap = t - last_end
        post[:, t] = np.where((~promo[:, t]) & (gap < 7), 1.0, 0.0)

    hol = cal["holiday"].fillna("").to_numpy()
    hol_types = {"": 0, "Tết Dương lịch": 1, "Tết Nguyên Đán": 2}
    hol_id = np.array([hol_types.get(h, 3) for h in hol])
    # normal days for level estimation
    tet_zone = (k >= -45) & (k <= 20)
    normal_day = ~tet_zone & (hol_id == 0)
    normal = exists & ~stockout & ~closed & (disc == 0) & (post == 0) & normal_day[None, :]

    # 5-day centred mean of clean sales, used for the last-season (season-aligned) lag
    clean = np.where(exists & ~stockout & ~closed, sales, np.nan)
    smooth = pd.DataFrame(clean.T).rolling(5, center=True, min_periods=2).mean().to_numpy().T
    smooth = smooth.astype(np.float32)

    first_day = np.where(exists.any(1), exists.argmax(1), D)
    it = items.set_index("item_id")
    st = stores.set_index("store_id")
    return dict(
        days=days, ser=ser, sales=sales, smooth=smooth, exists=exists, disc=disc, rowid=rowid, post=post,
        stockout=stockout, closed=closed, normal=normal, k=k, lny=lunar_new_year,
        dow=cal["day_of_week"].to_numpy(), lday=cal["lunar_day"].to_numpy(),
        lmon=cal["lunar_month"].to_numpy(), hol=hol_id, first=first_day,
        price=ser.item_id.map(it.regular_price).to_numpy(np.float32),
        cat=ser.item_id.map(it.category).to_numpy(), stype=ser.store_id.map(st.store_type).to_numpy(),
        city=ser.store_id.map(st.city).to_numpy(),
    )


def series_levels(P: dict, o: int) -> tuple[np.ndarray, ...]:
    """Baseline level per series at origin o (index of the last observed day)."""
    L, Sh = CFG["base_long"], CFG["base_short"]
    sales, normal = P["sales"], P["normal"]

    def mean_last(n: int) -> tuple[np.ndarray, np.ndarray]:
        # mean over the last n *normal* days before o (look back up to 150 days)
        lo = max(0, o - 150)
        m = normal[:, lo:o + 1]
        v = np.where(m, sales[:, lo:o + 1], 0.0)
        cm = np.cumsum(m[:, ::-1], 1)
        take = m[:, ::-1] & (cm <= n)
        cnt = take.sum(1)
        s = np.where(take, v[:, ::-1], 0).sum(1)
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan), cnt

    base, cnt = mean_last(L)
    short, _ = mean_last(Sh)
    age = o - P["first"]
    young = (age >= 0) & (age < CFG["young_days"])
    # young series (new store / new item, still ramping up): use the last 14 days only
    W = CFG["young_window"]
    v = np.where(P["exists"][:, o - W + 1:o + 1] & ~P["stockout"][:, o - W + 1:o + 1],
                 P["sales"][:, o - W + 1:o + 1], np.nan)
    with np.errstate(all="ignore"):
        recent = np.nanmean(v, 1)
    base = np.where(young & ~np.isnan(recent), recent, base)
    has = (cnt >= 7) | (young & ~np.isnan(recent))
    has &= ~P["ser"].item_id.isin(SEASONAL).to_numpy()
    base = np.where(has, base, np.nan)
    trend = np.where(has, np.log((short + 0.5) / (base + 0.5)), 0.0)
    trend = np.nan_to_num(np.clip(trend, -1, 1))
    return base, has, trend, young, age


# ----------------------------------------------------------------------------- windows
def make_window(P: dict, o: int, enc: dict, cutoff: int) -> dict | None:
    """All rows (series, day) with o < day <= min(o+H, cutoff) that exist."""
    hi = min(o + H, cutoff, len(P["days"]) - 1)
    if hi <= o:
        return None
    base, has, trend, young, age = series_levels(P, o)
    ex = P["exists"][:, o + 1:hi + 1]
    si, dj = np.nonzero(ex)
    t = dj + o + 1
    ser = P["ser"]
    # store scale: mean baseline revenue per item (robust to which items exist)
    rev = np.where(has, base * P["price"], np.nan)
    sdf = pd.DataFrame({"st": ser.store_id, "rev": rev, "it": ser.item_id, "b": base})
    item_ref = sdf.groupby("it").rev.transform("median")
    store_scale = (sdf.rev / item_ref).groupby(sdf.st).transform("median").fillna(1.0).to_numpy()
    item_mean = sdf.groupby("it").b.transform("mean").to_numpy()
    k = np.clip(P["k"][t], K_MIN - 1, K_MAX + 1) - (K_MIN - 1)
    item_cat = ser.item_id.map(enc["item"]).to_numpy()[si]
    cat = np.stack([
        ser.store_id.map(enc["store"]).to_numpy()[si], item_cat,
        pd.Series(P["cat"]).map(enc["cat"]).to_numpy()[si],
        pd.Series(P["stype"]).map(enc["stype"]).to_numpy()[si],
        pd.Series(P["city"]).map(enc["city"]).to_numpy()[si],
        P["dow"][t], k, P["lday"][t] - 1, P["hol"][t], np.minimum((t - o - 1) // 7, 9),
    ], 1).astype(np.int64)
    b = base[si]
    hb = has[si].astype(np.float32)
    num = np.stack([
        P["disc"][si, t] * 3, P["post"][si, t], np.log1p(np.nan_to_num(b)) / 3, hb,
        trend[si], np.log(store_scale[si]), np.log1p(np.nan_to_num(item_mean[si])) / 3,
        young[si].astype(np.float32), np.log1p(np.clip(age[si], 0, 400)) / 6,
    ], 1).astype(np.float32)
    # season-aligned lag: same store-item, same days-to-Tết (or same weekday 52 weeks ago) last season
    tet_days = np.array([(pd.Timestamp(v) - P["days"][0]).days for v in TET.values()])
    tet_this = t - P["k"][t]
    j = np.searchsorted(tet_days, tet_this)
    prev_tet = np.where(j > 0, tet_days[np.maximum(j - 1, 0)], tet_this - 364)
    shift = np.where(np.abs(P["k"][t]) <= 60, tet_this - prev_tet, 364)
    tp = t - shift
    okp = tp >= 0
    lys = np.where(okp, P["smooth"][si, np.maximum(tp, 0)], np.nan)
    base_p = series_levels(P, max(o - 364, 30))[0][si]
    has_ly = ~np.isnan(lys)
    ly_rel = np.where(has_ly & ~np.isnan(base_p) & (hb > 0),
                      np.log((np.nan_to_num(lys) + 0.3) / (np.nan_to_num(base_p) + 0.3)), 0.0)
    ly_abs = np.where(has_ly, np.log1p(np.nan_to_num(lys)) / 3, 0.0)
    num = np.concatenate([num, np.stack([np.clip(ly_rel, -3, 4), ly_abs, has_ly], 1).astype(np.float32)], 1)
    offset = np.where(hb > 0, np.log(np.maximum(np.nan_to_num(b), 0.05)), 0.0).astype(np.float32)
    y = P["sales"][si, t]
    ok = ~P["stockout"][si, t] & ~P["closed"][si, t] & ~P["lny"][t]
    return dict(cat=cat, num=num, off=offset, y=y, ok=ok, w=P["price"][si],
                si=si, t=t, rowid=P["rowid"][si, t], lny=P["lny"][t],
                so=P["stockout"][si, t] | P["closed"][si, t])


def origins_for(P: dict, cutoff: int, final_origin: int) -> list[int]:
    days = P["days"]
    res = []
    for y in (2023, 2024, 2025, 2026):
        o = (pd.Timestamp(f"{y - 1}-12-31") - days[0]).days
        if o < final_origin:
            res.append(o)
    if CFG["extra_origins"]:
        for mstart in pd.date_range("2023-03-01", "2025-12-01", freq="MS"):
            if mstart.month in (1, 12):
                continue
            o = (mstart - days[0]).days - 1
            if o + 14 < cutoff:
                res.append(o)
    return res


# ----------------------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, sizes: list[int], dims: list[int], n_num: int, hidden: int) -> None:
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, d) for n, d in zip(sizes, dims)])
        din = sum(dims) + n_num
        self.mlp = nn.Sequential(
            nn.Linear(din, hidden), nn.SiLU(), nn.Dropout(0.05),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, cat: torch.Tensor, num: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
        x = torch.cat([e(cat[:, j]) for j, e in enumerate(self.embs)] + [num], 1)
        return (off + self.mlp(x).squeeze(-1)).clamp(-8, 9)  # log mean


def train_predict(tr: dict, te: dict, sizes: list[int], seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dims = [4, 12, 4, 3, 2, 3, 12, 6, 3, 3]
    net = Net(sizes, dims, tr["num"].shape[1], CFG["hidden"])
    m = tr["ok"] & ~np.isnan(tr["y"])
    T = {k: torch.tensor(tr[k][m]) for k in ("cat", "num", "off", "y", "w")}
    T["w"] = T["w"] * torch.tensor(np.where(tr["tet_win"][m], CFG["tet_w"], 1.0), dtype=torch.float32)
    T["w"] = T["w"] / T["w"].mean()
    n = len(T["y"])
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    steps = CFG["epochs"] * ((n + CFG["batch"] - 1) // CFG["batch"])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["lr"], total_steps=steps, pct_start=0.15)
    for ep in range(CFG["epochs"]):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, CFG["batch"]):
            b = perm[i:i + CFG["batch"]]
            lm = net(T["cat"][b], T["num"][b], T["off"][b])
            loss = (T["w"][b] * (torch.exp(lm) - T["y"][b] * lm)).mean()  # weighted Poisson
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            tot += loss.item() * len(b)
    net.eval()
    with torch.no_grad():
        lm = net(torch.tensor(te["cat"]), torch.tensor(te["num"]), torch.tensor(te["off"]))
    return np.exp(lm.numpy())


def wape(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * np.abs(y - p)) / np.sum(w * y))


# ----------------------------------------------------------------------------- post-processing
def i48_fix(P: dict, te: dict, enc: dict, o: int) -> None:
    """Cold start for the new gift box I48: it is predicted as the analog I16 (same category,
    similar price) and rescaled by I48/I16 sales at the same days-to-Tết in earlier seasons."""
    ser = P["ser"]
    i48 = ser.item_id.to_numpy()[te["si"]] == "I48"
    if not i48.any():
        return
    ratio = CFG["i48_ratio"]
    if ratio is None:
        m48 = (ser.item_id == "I48").to_numpy()
        m16 = (ser.item_id == "I16").to_numpy()
        hist = np.arange(o + 1)
        kk = P["k"][hist]
        win = hist[(kk >= -56) & (kk <= -45)]
        a = np.nanmean(np.where(P["exists"][m48][:, win], P["sales"][m48][:, win], np.nan))
        # I16 at the same k in past seasons
        past = np.arange(o + 1)
        pw = past[(P["k"][past] >= -49) & (P["k"][past] <= -42)]
        bvals = np.where(P["exists"][m16][:, pw], P["sales"][m16][:, pw], np.nan)
        bb = np.nanmean(bvals)
        ratio = float(a / bb) if np.isfinite(a) and np.isfinite(bb) and bb > 0 else 0.6
    te["i48_ratio"] = ratio
    te["pred"][i48] *= ratio


def young_growth(P: dict, te: dict, o: int) -> None:
    """Ramp-up extrapolation for young series (new store S10, new item I47): fit a saturating
    curve L(t) = A * (1 - exp(-t / tau)) to weekly levels and scale the forecast."""
    ser = P["ser"]
    for kind, key in (("store", "store_id"), ("item", "item_id")):
        for name in ser[key].unique():
            m = (ser[key] == name).to_numpy()
            first = P["first"][m].min()
            age = o - first
            if age < 28 or age > 150:
                continue
            ex = P["exists"][m][:, first:o + 1]
            v = np.where(ex & ~P["stockout"][m][:, first:o + 1], P["sales"][m][:, first:o + 1], 0)
            v = v * P["price"][m][:, None] if kind == "store" else v
            daily = v.sum(0) / np.maximum(ex.sum(0), 1) if kind == "item" else v.sum(0)
            nw = len(daily) // 7
            wk = daily[len(daily) - nw * 7:].reshape(nw, 7).mean(1)
            tw = np.arange(nw) + 0.5 + (len(daily) - nw * 7) / 7
            best = None
            for tau in np.linspace(1, 40, 80):
                f = 1 - np.exp(-tw / tau)
                A = (f @ wk) / (f @ f)
                err = ((A * f - wk) ** 2).sum()
                if best is None or err < best[0]:
                    best = (err, A, tau)
            _, A, tau = best
            last2 = wk[-2:].mean()
            th = tw[-1] + 1 + np.arange(0, H / 7)
            fut = A * (1 - np.exp(-th / tau))
            g = float(np.clip(fut.mean() / max(last2, 1e-6), 1.0, 1.25))
            sel = m[te["si"]]
            te["pred"][sel] *= g
            te.setdefault("growth", {})[name] = round(g, 3)


# ----------------------------------------------------------------------------- driver
def prepare(target_year: int, P: dict) -> dict:
    """Build training windows (all origins whose targets end <= the forecast origin) + the test window."""
    days = P["days"]
    o = (pd.Timestamp(f"{target_year - 1}-12-31") - days[0]).days
    cutoff = o  # no target day after the origin may be used for training
    ser = P["ser"]
    enc = dict(
        store={s: i for i, s in enumerate(sorted(ser.store_id.unique()))},
        item={s: i for i, s in enumerate(sorted(ser.item_id.unique()))},
        cat={s: i for i, s in enumerate(sorted(set(P["cat"])))},
        stype={s: i for i, s in enumerate(sorted(set(P["stype"])))},
        city={s: i for i, s in enumerate(sorted(set(P["city"])))},
    )
    enc["item"] = {**enc["item"], "I48": enc["item"]["I16"]}  # cold-start analog
    sizes = [len(set(enc["store"].values())), len(set(enc["item"].values())) + 1, len(enc["cat"]),
             len(enc["stype"]), len(enc["city"]), 7, K_MAX - K_MIN + 3, 30, 4, 10]
    orig = origins_for(P, cutoff, o)
    ws = [(oo, make_window(P, oo, enc, cutoff)) for oo in orig]
    ws = [(oo, w) for oo, w in ws if w is not None]
    keys = ("cat", "num", "off", "y", "ok", "w")
    tr = {k: np.concatenate([w[k] for _, w in ws]) for k in keys}
    tr["tet_win"] = np.concatenate([np.full(len(w["y"]), days[oo].month == 12) for oo, w in ws])
    te = make_window(P, o, enc, len(days) - 1)
    return dict(tr=tr, te=te, sizes=sizes, o=o, enc=enc, P=P, year=target_year)


def fit_predict(D: dict, verbose: bool = True) -> dict:
    t0 = time.time()
    tr, te, P, o = D["tr"], dict(D["te"]), D["P"], D["o"]
    if not CFG["extra_origins"]:
        tr = {k: v[tr["tet_win"]] for k, v in tr.items()}
    if not CFG["use_ly"]:  # ablation: drop the season-aligned lag features
        tr = {**tr, "num": tr["num"][:, :9]}
        te = {**te, "num": te["num"][:, :9]}
    preds = [train_predict(tr, te, D["sizes"], seed) for seed in range(CFG["seeds"])]
    te["pred"] = np.mean(preds, 0) * CFG["calib"]
    te["pred"][te["lny"]] = 0.0  # stores are closed on lunar New Year's day
    i48_fix(P, te, D["enc"], o)
    if CFG["young_growth"]:
        young_growth(P, te, o)
    if verbose:
        print(f"[{D['year']}] train rows={len(tr['y']):,} fit {time.time() - t0:.0f}s", flush=True)
    te["P"] = P
    return te


def run(target_year: int, P: dict | None = None) -> dict:
    if P is None:
        P = build_panel(load())
    return fit_predict(prepare(target_year, P))


def report(te: dict) -> dict:
    P = te["P"]
    m = ~np.isnan(te["y"]) & ~te["so"]  # test period has no stock-outs
    st = P["ser"].store_id.to_numpy()[te["si"]]
    pub = np.isin(st, ["S01", "S02", "S04", "S06", "S09"])
    out = {"all": wape(te["y"][m], te["pred"][m], te["w"][m]),
           "pubstores": wape(te["y"][m & pub], te["pred"][m & pub], te["w"][m & pub]),
           "privstores": wape(te["y"][m & ~pub], te["pred"][m & ~pub], te["w"][m & ~pub])}
    return {k: round(v, 4) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", type=int, default=0)
    args = ap.parse_args()
    if args.backtest:
        te = run(args.backtest)
        print("backtest", args.backtest, report(te), te.get("growth"), te.get("i48_ratio"))
        return
    te = run(2026)
    print("growth factors:", te.get("growth"), "I48 ratio:", te.get("i48_ratio"))
    pred = pd.Series(te["pred"], index=te["rowid"])
    for part in ("public", "private"):
        ids = pd.read_csv(f"{DATA}/{part}_test/{part}_test.csv")["id"]
        sub = pd.DataFrame({"id": ids, "sales": np.clip(pred.reindex(ids).to_numpy(), 0, None)})
        assert sub["sales"].notna().all()
        sub.to_csv(os.path.join(HERE, f"{part}_submission.csv"), index=False)
        print("wrote", part, len(sub), "rows, total", round(sub.sales.sum()))


if __name__ == "__main__":
    main()
