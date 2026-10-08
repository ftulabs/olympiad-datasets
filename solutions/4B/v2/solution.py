"""4B - Tết demand forecasting, v2. End-to-end solution.

Direct multi-horizon framing, aligned on Tết: every training example is one (store, item, target day)
inside a 63-day window after an origin day. All level features come from history up to the origin.
Final forecast: origin 2025-12-31 (= Tết 2026 - 48 days), window 2026-01-01 .. 2026-03-04.

Two PyTorch model families, blended on out-of-fold backtests:
  * MLP  : embeddings + numeric features, output = log(level) + MLP(x)   (v1 family, stronger features)
  * GLM  : log-additive factor model with crossed scalar embeddings
           (item x days-to-Tết, store x days-to-Tết, store-type x weekday, category x lunar day, promo ...)
Both are trained with a revenue-weighted Poisson loss. The final blend is a weighted geometric mean.
Post-processing then turns the mean forecast into the median of a negative-binomial predictive
distribution, which is the WAPE-optimal point forecast.

Usage:
  python solution.py                          # final fit on all history, writes the two CSVs
  python solution.py --fold V25 --model mlp   # backtest (origin 2024-12-12), saves OOF preds
  python solution.py --cfg '{"seeds": 3}'     # JSON overrides of CFG
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import nbinom

HERE = os.path.dirname(os.path.abspath(globals().get("__file__", "solution.py")))
DATA = os.environ.get("DATA_4B", "/home/minh/Desktop/olympiad_ai/warmup/4B_tet_demand_forecast/dataset")
H = 63  # 1 Jan .. 4 Mar
TET = {2023: "2023-01-22", 2024: "2024-02-10", 2025: "2025-01-29", 2026: "2026-02-17"}
K_MIN, K_MAX = -70, 45
SEASONAL = ["I15", "I16", "I30", "I48"]
PUB_STORES = ["S01", "S02", "S04", "S06", "S09"]
# validation folds: (origin, target Tết year). V* are aligned on Tết like the test (origin = Tết - 48)
FOLDS = {"V25": ("2024-12-12", 2025), "V24": ("2023-12-24", 2024),
         "D25": ("2024-12-31", 2025), "D24": ("2023-12-31", 2024), "FINAL": ("2025-12-31", 2026)}
CFG = dict(
    base_long=56, base_short=28, young_days=60, young_window=14,
    epochs=12, lr=6e-3, batch=4096, hidden=192, layers=2, dropout=0.05, seeds=5, wd=1e-5,
    jitter=[-21, -14, -7, 0, 7], dec31=True, generic=True, generic_step="MS", tet_w=1.0,
    extra_levels=True, i48_ratio=None, young_growth=True, calib=1.0,
    glm_epochs=10, glm_lr=0.03, glm_l2=3e-6, glm_dk=3e-4, glm_batch=8192,
    nb_phi=0.0, nb_q=0.5, threads=1, device="auto", store_analog={},
)


def device() -> torch.device:
    if CFG["device"] != "auto":
        return torch.device(CFG["device"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- data
def load() -> dict:
    tr = pd.read_csv(f"{DATA}/train/train.csv", parse_dates=["date"])
    pu = pd.read_csv(f"{DATA}/public_test/public_test.csv", parse_dates=["date"])
    pr = pd.read_csv(f"{DATA}/private_test/private_test.csv", parse_dates=["date"])
    items = pd.read_csv(f"{DATA}/train/items.csv")
    stores = pd.read_csv(f"{DATA}/train/stores.csv")
    cal = pd.read_csv(f"{DATA}/train/calendar.csv", parse_dates=["date"])
    df = pd.concat([tr, pu.drop(columns="weight"), pr.drop(columns="weight")], ignore_index=True)
    return dict(df=df, items=items, stores=stores, cal=cal)


def build_panel(d: dict) -> dict:
    """Dense [series x day] arrays: sales, exists, discount, normal-day mask, stock-out mask."""
    df, items, stores, cal = d["df"], d["items"], d["stores"], d["cal"]
    days = pd.date_range("2022-11-01", "2026-03-04")
    D = len(days)
    ser = df[["store_id", "item_id"]].drop_duplicates().sort_values(["store_id", "item_id"]).reset_index(drop=True)
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
    if CFG.get("sim_new"):  # validation only: pretend a store opened on a given date (mimics S10)
        st_new, d_new = CFG["sim_new"]
        cut = (pd.Timestamp(d_new) - days[0]).days
        msk = (ser.store_id == st_new).to_numpy()
        sales[np.ix_(msk, np.arange(cut))] = np.nan
        exists[np.ix_(msk, np.arange(cut))] = False
    disc[r, c] = df["discount_pct"].to_numpy(np.float32)
    rowid[r, c] = df["id"].to_numpy()

    cal = cal.set_index("date").reindex(days)
    tets = np.array([(pd.Timestamp(v) - days[0]).days for v in TET.values()])
    dn = np.arange(D)
    near = tets[np.abs(dn[:, None] - tets[None, :]).argmin(1)]
    k = dn - near
    lny = (cal["lunar_month"].to_numpy() == 1) & (cal["lunar_day"].to_numpy() == 1)
    last_train = (pd.Timestamp("2025-12-31") - days[0]).days

    closed = np.zeros((S, D), bool)  # store closed: all items 0 (Tết day, S05 renovation)
    for st in ser.store_id.unique():
        m = (ser.store_id == st).to_numpy()
        stot = np.nansum(np.where(exists[m], sales[m], 0), 0)
        has = exists[m].any(0) & (dn <= last_train)
        closed[m] = (has & (stot == 0))[None, :]

    sdf = pd.DataFrame(np.where(closed, np.nan, sales).T)
    med = sdf.rolling(29, center=True, min_periods=7).median().to_numpy().T
    zero = (sales == 0) & ~closed
    prev0 = np.zeros_like(zero)
    prev0[:, 1:] = zero[:, :-1]
    next0 = np.zeros_like(zero)
    next0[:, :-1] = zero[:, 1:]
    stockout = zero & ((med >= 4) | ((prev0 | next0) & (med >= 2)))

    promo = disc > 0
    post = np.zeros((S, D), np.float32)
    last_end = np.full(S, -999)
    for t in range(1, D):
        ended = promo[:, t - 1] & ~promo[:, t]
        last_end = np.where(ended, t, last_end)
        post[:, t] = np.where((~promo[:, t]) & (t - last_end < 7), 1.0, 0.0)

    hol = cal["holiday"].fillna("").to_numpy()
    hol_types = {"": 0, "Tết Dương lịch": 1, "Tết Nguyên Đán": 2}
    hol_id = np.array([hol_types.get(h, 3) for h in hol])
    tet_zone = (k >= -45) & (k <= 20)
    normal_day = ~tet_zone & (hol_id == 0)
    normal = exists & ~stockout & ~closed & (disc == 0) & (post == 0) & normal_day[None, :]

    first_day = np.where(exists.any(1), exists.argmax(1), D)
    # true birth of a series: store opening date or first sale of a newly launched item (series present
    # at the start of the data are "old", not young)
    opened = ser.store_id.map(stores.set_index("store_id").open_date).pipe(pd.to_datetime)
    open_idx = ((opened - days[0]).dt.days).to_numpy()
    born = np.where((first_day > 30) | (open_idx > 0), np.maximum(first_day, open_idx), -10 ** 4)
    it = items.set_index("item_id")
    st = stores.set_index("store_id")
    return dict(
        days=days, ser=ser, sales=sales, exists=exists, disc=disc, rowid=rowid, post=post,
        stockout=stockout, closed=closed, normal=normal, k=k, lny=lny,
        dow=cal["day_of_week"].to_numpy(), lday=cal["lunar_day"].to_numpy(), hol=hol_id, first=first_day,
        born=born if CFG.get("true_age", True) else first_day,
        price=ser.item_id.map(it.regular_price).to_numpy(np.float32),
        cat=ser.item_id.map(it.category).to_numpy(), stype=ser.store_id.map(st.store_type).to_numpy(),
        city=ser.store_id.map(st.city).to_numpy(),
    )


def mean_last(P: dict, o: int, n: int, look: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Mean over the last n *normal* days up to o (looking back at most `look` days)."""
    lo = max(0, o - look)
    m = P["normal"][:, lo:o + 1][:, ::-1]
    v = np.where(m, P["sales"][:, lo:o + 1][:, ::-1], 0.0)
    take = m & (np.cumsum(m, 1) <= n)
    cnt = take.sum(1)
    s = np.where(take, v, 0).sum(1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan), cnt


def series_levels(P: dict, o: int) -> dict:
    """Baseline level per series at origin o plus auxiliary level features."""
    base, cnt = mean_last(P, o, CFG["base_long"])
    short, _ = mean_last(P, o, CFG["base_short"])
    vshort, _ = mean_last(P, o, 14)
    longb, _ = mean_last(P, o, 112, 300)
    age = np.clip(o - P["born"], -1, 400)
    young = (age >= 0) & (age < CFG["young_days"])
    W = CFG["young_window"]
    sl = slice(o - W + 1, o + 1)
    v = np.where(P["exists"][:, sl] & ~P["stockout"][:, sl], P["sales"][:, sl], np.nan)
    with np.errstate(all="ignore"):
        recent = np.nanmean(v, 1)
    base = np.where(young & ~np.isnan(recent), recent, base)
    has = (cnt >= 7) | (young & ~np.isnan(recent))
    has &= ~P["ser"].item_id.isin(SEASONAL).to_numpy()
    base = np.where(has, base, np.nan)

    def rel(x: np.ndarray) -> np.ndarray:
        r = np.where(has, np.log((np.nan_to_num(x, nan=0) + 0.5) / (np.nan_to_num(base) + 0.5)), 0.0)
        return np.clip(np.where(np.isnan(x), 0, r), -1.5, 1.5)

    # pooled level: long-run series level x recent store factor x recent item factor (less noisy)
    yearb, ycnt = mean_last(P, o, 364, 420)
    ser = P["ser"]
    ok = has & (ycnt >= 60) & (yearb > 0.3)
    lr = pd.Series(np.where(ok, np.log((np.nan_to_num(base) + 0.3) / (np.nan_to_num(yearb) + 0.3)), np.nan))
    fs = lr.groupby(ser.store_id.to_numpy()).transform("median")
    fi = (lr - fs).groupby(ser.item_id.to_numpy()).transform("median")
    pooled = np.exp(np.log(np.nan_to_num(yearb) + 0.3) + fs.fillna(0).to_numpy() + fi.fillna(0).to_numpy()) - 0.3
    pool = np.where(ok, np.log((np.maximum(pooled, 0) + 0.3) / (np.nan_to_num(base) + 0.3)), 0.0)
    # a ramping-up series (new store / new item) has trend features distorted by its ramp: neutralise
    keep = (age >= CFG.get("trend_min_age", 150)) | (not CFG.get("true_age", True))
    z = lambda x: np.where(keep, x, 0.0)  # noqa: E731
    return dict(base=base, has=has, trend=z(rel(short)), vtrend=z(rel(vshort)), ltrend=z(rel(longb)),
                young=young, age=age, pool=z(np.clip(pool, -1.5, 1.5)), lowvol=(np.nan_to_num(base) < 3) & has)


# ----------------------------------------------------------------------------- windows
def make_window(P: dict, o: int, enc: dict, cutoff: int) -> dict | None:
    """All rows (series, day) with o < day <= min(o+H, cutoff) that exist."""
    hi = min(o + H, cutoff, len(P["days"]) - 1)
    if hi <= o:
        return None
    L = series_levels(P, o)
    base, has = L["base"], L["has"]
    ex = P["exists"][:, o + 1:hi + 1]
    si, dj = np.nonzero(ex)
    t = dj + o + 1
    ser = P["ser"]
    rev = np.where(has, base * P["price"], np.nan)
    sdf = pd.DataFrame({"st": ser.store_id, "rev": rev, "it": ser.item_id, "b": base})
    item_ref = sdf.groupby("it").rev.transform("median")
    store_scale = (sdf.rev / item_ref).groupby(sdf.st).transform("median").fillna(1.0).to_numpy()
    item_mean = sdf.groupby("it").b.transform("mean").to_numpy()
    k = np.clip(P["k"][t], K_MIN - 1, K_MAX + 1) - (K_MIN - 1)
    st_true = ser.store_id.map(enc["store_true"]).to_numpy()[si]
    cat = np.stack([
        ser.store_id.map(enc["store"]).to_numpy()[si], ser.item_id.map(enc["item"]).to_numpy()[si],
        pd.Series(P["cat"]).map(enc["cat"]).to_numpy()[si],
        pd.Series(P["stype"]).map(enc["stype"]).to_numpy()[si],
        pd.Series(P["city"]).map(enc["city"]).to_numpy()[si],
        P["dow"][t], k, P["lday"][t] - 1, P["hol"][t], np.minimum((t - o - 1) // 7, 9),
    ], 1).astype(np.int64)
    b = base[si]
    hb = has[si].astype(np.float32)
    feats = [
        P["disc"][si, t] * 3, P["post"][si, t], np.log1p(np.nan_to_num(b)) / 3, hb,
        L["trend"][si], np.log(store_scale[si]), np.log1p(np.nan_to_num(item_mean[si])) / 3,
        L["young"][si].astype(np.float32), np.log1p(np.clip(L["age"][si], 0, 400)) / 6,
    ]
    if CFG["extra_levels"]:
        feats += [L["vtrend"][si], L["ltrend"][si], (t - o) / H]
    if CFG.get("pool", True):
        feats += [L["pool"][si], L["pool"][si] * L["lowvol"][si], L["pool"][si] / np.sqrt(1 + np.nan_to_num(b))]
    num = np.stack(feats, 1).astype(np.float32)
    offset = np.where(hb > 0, np.log(np.maximum(np.nan_to_num(b), 0.05)), 0.0).astype(np.float32)
    y = P["sales"][si, t]
    ok = ~P["stockout"][si, t] & ~P["closed"][si, t] & ~P["lny"][t]
    return dict(cat=cat, num=num, off=offset, y=y, ok=ok, w=P["price"][si], disc=P["disc"][si, t], st=st_true,
                post=P["post"][si, t], si=si, t=t, rowid=P["rowid"][si, t], lny=P["lny"][t],
                so=P["stockout"][si, t] | P["closed"][si, t])


def origins_for(P: dict, o: int) -> list[tuple[int, bool]]:
    """Training origins (origin index, is_tet_window); every target day must be <= o."""
    days = P["days"]
    res = []
    for y, tdate in TET.items():
        T = (pd.Timestamp(tdate) - days[0]).days
        if T >= o:
            continue
        for j in CFG["jitter"]:
            oo = T - 48 + j
            if oo >= 28:
                res.append((oo, True))
        if CFG["dec31"]:
            oo = (pd.Timestamp(f"{y - 1}-12-31") - days[0]).days
            if oo >= 28:
                res.append((oo, True))
    if CFG["generic"]:
        for mstart in pd.date_range("2023-03-01", "2025-12-01", freq=CFG["generic_step"]):
            if mstart.month in (1, 12):
                continue
            oo = (mstart - days[0]).days - 1
            if oo + 14 < o:
                res.append((oo, False))
    return sorted(set(res))


# ----------------------------------------------------------------------------- models
class Net(nn.Module):
    """v1-style MLP on embeddings; predicts log multiplier on top of log(level)."""

    def __init__(self, sizes: list[int], dims: list[int], n_num: int, hidden: int, layers: int, drop: float) -> None:
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, d) for n, d in zip(sizes, dims)])
        din = sum(dims) + n_num
        mods: list[nn.Module] = []
        for _ in range(layers):
            mods += [nn.Linear(din, hidden), nn.SiLU(), nn.Dropout(drop)]
            din = hidden
        self.mlp = nn.Sequential(*mods, nn.Linear(din, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, cat: torch.Tensor, num: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
        x = torch.cat([e(cat[:, j]) for j, e in enumerate(self.embs)] + [num], 1)
        return (off + self.mlp(x).squeeze(-1)).clamp(-8, 9)


# crossed scalar terms of the GLM: (name, column tuple). Columns of `cat`:
# 0 store 1 item 2 cat 3 stype 4 city 5 dow 6 k 7 lday 8 hol 9 hweek ; 10 discbucket 11 post 12 hasbase 13 young
GLM_TERMS = [
    ("item_k", (1, 6)), ("cat_k", (2, 6)), ("stype_k", (3, 6)), ("store_k", (0, 6)), ("city_k", (4, 6)),
    ("stype_dow", (3, 5)), ("store_dow", (0, 5)), ("cat_dow", (2, 5)), ("item_dow", (1, 5)),
    ("cat_lday", (2, 7)), ("item_lday", (1, 7)), ("cat_hol", (2, 8)), ("stype_hol", (3, 8)),
    ("cat_disc", (2, 10)), ("item_disc", (1, 10)), ("cat_post", (2, 11)),
    ("store_item", (14, 1)), ("has", (12,)), ("item_has", (1, 12)),
    ("store_hw", (0, 9)), ("item_hw", (1, 9)), ("young_hw", (13, 9)), ("stype_dowk", (3, 5, 12)),
]
K_TERMS = {"item_k", "cat_k", "stype_k", "store_k", "city_k", "cat_stype_k"}
GLM_EXTRA = [("cat_stype_k", (2, 3, 6)), ("item_kw", (1, 15)), ("cat_kw_stype", (2, 15, 3)),
             ("stype_hw", (3, 9)), ("cat_hw", (2, 9))]


class GLM(nn.Module):
    def __init__(self, card: list[int], n_num: int) -> None:
        super().__init__()
        self.terms = GLM_TERMS + (GLM_EXTRA if CFG.get("glm_extra") else [])
        self.card = card
        self.tabs = nn.ModuleList()
        for _, cols in self.terms:
            n = int(np.prod([card[c] for c in cols]))
            e = nn.Embedding(n, 1)
            nn.init.zeros_(e.weight)
            self.tabs.append(e)
        self.lin = nn.Linear(n_num, 1)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def index(self, X: torch.Tensor, cols: tuple) -> torch.Tensor:
        idx = X[:, cols[0]]
        for c in cols[1:]:
            idx = idx * self.card[c] + X[:, c]
        return idx

    def forward(self, X: torch.Tensor, num: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
        s = off + self.lin(num).squeeze(-1)
        for (_, cols), e in zip(self.terms, self.tabs):
            s = s + e(self.index(X, cols)).squeeze(-1)
        return s.clamp(-8, 9)

    def penalty(self, l2: float, dk: float) -> torch.Tensor:
        p = 0.0
        nk = self.card[6]
        for (name, cols), e in zip(self.terms, self.tabs):
            w = e.weight.squeeze(-1)
            p = p + l2 * (w ** 2).sum()
            if name in K_TERMS:
                wk = w.view(-1, nk) if name != "cat_stype_k" else w.view(-1, nk)
                p = p + l2 * CFG.get("glm_fine", 0.0) * (w ** 2).sum()
                p = p + dk * ((wk[:, 1:] - wk[:, :-1]) ** 2).sum()
        return p


def glm_X(d: dict) -> np.ndarray:
    dbk = np.searchsorted(np.array([0.01, 0.12, 0.17, 0.22, 0.27]), d["disc"])
    hasb = (d["off"] != 0).astype(np.int64)
    young = (d["num"][:, 7] > 0).astype(np.int64)
    kweek = d["cat"][:, 6] // 7
    return np.concatenate([d["cat"], np.stack([dbk, d["post"].astype(np.int64), hasb, young, d["st"], kweek], 1)], 1)


def train_predict(tr: dict, te: dict, sizes: list[int], seed: int, kind: str) -> np.ndarray:
    if kind == "hyb":  # GLM first, then an MLP learns the residual on top of the GLM log-prediction
        g_tr, g_te = _train_predict(tr, te, sizes, seed, "glm", both=True)
        ep = CFG["epochs"]
        CFG["epochs"] = CFG.get("hyb_epochs", 6)
        try:
            return train_predict({**tr, "off": g_tr}, {**te, "off": g_te}, sizes, seed, "mlp")
        finally:
            CFG["epochs"] = ep
    return _train_predict(tr, te, sizes, seed, kind)


def _train_predict(tr: dict, te: dict, sizes: list[int], seed: int, kind: str, both: bool = False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    m = tr["ok"] & ~np.isnan(tr["y"])
    if kind == "glm":
        card = sizes + [6, 2, 2, 2, sizes[0], sizes[6] // 7 + 1]
        Xtr, Xte = glm_X(tr), glm_X(te)
        net = GLM(card, tr["num"].shape[1])
        T = dict(cat=torch.tensor(Xtr[m]), num=torch.tensor(tr["num"][m]), off=torch.tensor(tr["off"][m]),
                 y=torch.tensor(tr["y"][m]), w=torch.tensor(tr["w"][m]))
        E = dict(cat=torch.tensor(Xte), num=torch.tensor(te["num"]), off=torch.tensor(te["off"]))
        epochs, lr, bs = CFG["glm_epochs"], CFG["glm_lr"], CFG["glm_batch"]
    else:
        dims = [4, 12, 4, 3, 2, 3, 12, 6, 3, 3]
        net = Net(sizes, dims, tr["num"].shape[1], CFG["hidden"], CFG["layers"], CFG["dropout"])
        T = {k: torch.tensor(tr[k][m]) for k in ("cat", "num", "off", "y", "w")}
        E = dict(cat=torch.tensor(te["cat"]), num=torch.tensor(te["num"]), off=torch.tensor(te["off"]))
        epochs, lr, bs = CFG["epochs"], CFG["lr"], CFG["batch"]
    T["w"] = T["w"] * torch.tensor(np.where(tr["tet_win"][m], CFG["tet_w"], 1.0), dtype=torch.float32)
    T["w"] = T["w"] / T["w"].mean()
    net = net.to(dev)
    T = {k: v.to(dev) for k, v in T.items()}
    n = len(T["y"])
    nb = (n + bs - 1) // bs
    if kind == "glm":
        opt = torch.optim.Adam(net.parameters(), lr=lr)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=CFG["wd"])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * nb, pct_start=0.15)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            lm = net(T["cat"][b], T["num"][b], T["off"][b])
            loss = (T["w"][b] * (torch.exp(lm) - T["y"][b] * lm)).mean()
            if kind == "glm":
                loss = loss + net.penalty(CFG["glm_l2"], CFG["glm_dk"]) * len(b) / n
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
    net.eval()
    with torch.no_grad():
        lm = net(*(E[k].to(dev) for k in ("cat", "num", "off")))
        if both:  # log-predictions for every training row (used as offset by the hybrid)
            Xall = torch.tensor(glm_X(tr) if kind == "glm" else tr["cat"])
            ltr = torch.cat([net(Xall[i:i + 65536].to(dev), torch.tensor(tr["num"][i:i + 65536]).to(dev),
                                 torch.tensor(tr["off"][i:i + 65536]).to(dev)) for i in range(0, len(Xall), 65536)])
            return ltr.cpu().numpy().astype(np.float32), lm.cpu().numpy().astype(np.float32)
    return np.exp(lm.cpu().numpy())


def wape(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * np.abs(y - p)) / np.sum(w * y))


# ----------------------------------------------------------------------------- post-processing
def i48_ratio(P: dict, o: int) -> float:
    """I48 (new 2026 gift box) is predicted as its analog I16 and rescaled by I48/I16 sales at the
    same days-to-Tết (I48's December 2025 vs I16's earlier seasons)."""
    if CFG["i48_ratio"] is not None:
        return CFG["i48_ratio"]
    ser = P["ser"]
    m48, m16 = (ser.item_id == "I48").to_numpy(), (ser.item_id == "I16").to_numpy()
    hist = np.arange(o + 1)
    kk = P["k"][hist]
    win = hist[(kk >= -56) & (kk <= -45)]
    a = np.nanmean(np.where(P["exists"][m48][:, win], P["sales"][m48][:, win], np.nan)) if m48.any() else np.nan
    pw = hist[(kk >= -49) & (kk <= -42)]
    bb = np.nanmean(np.where(P["exists"][m16][:, pw], P["sales"][m16][:, pw], np.nan))
    return float(a / bb) if np.isfinite(a) and np.isfinite(bb) and bb > 0 else 0.6


def young_growth(P: dict, si: np.ndarray, o: int) -> np.ndarray:
    """Ramp-up extrapolation for young stores/items: saturating curve fitted on weekly levels."""
    ser = P["ser"]
    g_all = np.ones(len(si))
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
            th = tw[-1] + 1 + np.arange(0, H / 7)
            g = float(np.clip((A * (1 - np.exp(-th / tau))).mean() / max(wk[-2:].mean(), 1e-6), 1.0, 1.25))
            g_all[m[si]] *= g
    return g_all


def postprocess(P: dict, te: dict, raw: np.ndarray, o: int) -> np.ndarray:
    """Business rules applied to a mean forecast (before the median step)."""
    pred = raw * CFG["calib"]
    pred = np.where(te["lny"], 0.0, pred)
    i48 = P["ser"].item_id.to_numpy()[te["si"]] == "I48"
    pred = np.where(i48, pred * i48_ratio(P, o), pred)
    if CFG["young_growth"]:
        pred = pred * young_growth(P, te["si"], o)
    return pred


def nb_median(mu: np.ndarray, phi: float, q: float = 0.5) -> np.ndarray:
    """q-quantile of NegBin(mean mu, var mu + phi mu^2) (Poisson when phi = 0)."""
    mu = np.maximum(mu, 1e-6)
    if phi <= 0:
        from scipy.stats import poisson
        return poisson.ppf(q, mu)
    r = 1.0 / phi
    return nbinom.ppf(q, r, r / (r + mu))


# ----------------------------------------------------------------------------- driver
def prepare(fold: str, P: dict) -> dict:
    days = P["days"]
    odate, year = FOLDS[fold]
    o = (pd.Timestamp(odate) - days[0]).days
    ser = P["ser"]
    enc = dict(
        store={s: i for i, s in enumerate(sorted(ser.store_id.unique()))},
        item={s: i for i, s in enumerate(sorted(ser.item_id.unique()))},
        cat={s: i for i, s in enumerate(sorted(set(P["cat"])))},
        stype={s: i for i, s in enumerate(sorted(set(P["stype"])))},
        city={s: i for i, s in enumerate(sorted(set(P["city"])))},
    )
    enc["item"] = {**enc["item"], "I48": enc["item"]["I16"]}  # cold-start analog
    for a, b in CFG.get("item_analog", {"I47": "I11"}).items():  # new item borrows an analog's interactions
        enc["item"][a] = enc["item"][b]
    enc["store_true"] = dict(enc["store"])
    for a, b in CFG.get("store_analog", {}).items():  # new store borrows an analog's interactions
        if a in enc["store"] and b in enc["store"]:
            enc["store"][a] = enc["store"][b]
    sizes = [len(enc["store_true"]), len(set(enc["item"].values())) + 1, len(enc["cat"]),
             len(enc["stype"]), len(enc["city"]), 7, K_MAX - K_MIN + 3, 30, 4, 10]
    ws = [(oo, tw, make_window(P, oo, enc, o)) for oo, tw in origins_for(P, o)]
    ws = [(oo, tw, w) for oo, tw, w in ws if w is not None]
    keys = ("cat", "num", "off", "y", "ok", "w", "disc", "post", "st")
    tr = {k: np.concatenate([w[k] for _, _, w in ws]) for k in keys}
    tr["tet_win"] = np.concatenate([np.full(len(w["y"]), tw) for _, tw, w in ws])
    if CFG.get("drop_nobase", True):  # rows of non-seasonal series with no level never occur at test time
        seas = np.isin(tr["cat"][:, 1], [enc["item"][i] for i in SEASONAL])
        tr["ok"] = tr["ok"] & ((tr["num"][:, 3] > 0) | seas)
    te = make_window(P, o, enc, len(days) - 1)
    return dict(tr=tr, te=te, sizes=sizes, o=o, enc=enc, P=P, fold=fold)


def fit_raw(D: dict, kind: str, verbose: bool = True) -> np.ndarray:
    t0 = time.time()
    s0 = CFG.get("seed0", 0)
    preds = [train_predict(D["tr"], D["te"], D["sizes"], seed, kind) for seed in range(s0, s0 + CFG["seeds"])]
    if verbose:
        print(f"[{D['fold']}/{kind}] train rows={len(D['tr']['y']):,} seeds={CFG['seeds']} "
              f"fit {time.time() - t0:.0f}s", flush=True)
    return np.exp(np.mean(np.log(np.maximum(preds, 1e-6)), 0)) if CFG.get("geo") else np.mean(preds, 0)


def report(P: dict, te: dict, pred: np.ndarray) -> dict:
    m = ~np.isnan(te["y"]) & ~te["so"]
    st = P["ser"].store_id.to_numpy()[te["si"]]
    pub = np.isin(st, PUB_STORES)
    out = {"all": wape(te["y"][m], pred[m], te["w"][m]),
           "pub": wape(te["y"][m & pub], pred[m & pub], te["w"][m & pub]),
           "priv": wape(te["y"][m & ~pub], pred[m & ~pub], te["w"][m & ~pub])}
    return {k: round(v, 4) for k, v in out.items()}


def blend(preds: list[np.ndarray], weights: list[float]) -> np.ndarray:
    lw = sum(w * np.log(np.maximum(p, 1e-6)) for p, w in zip(preds, weights)) / sum(weights)
    return np.exp(lw)


# final ensemble: (name, model kind, CFG overrides, blend weight) -- chosen on the V25/V24 backtests
FINAL_MODELS = [("mlp", "mlp", {"seeds": 6, "wd": 1e-2, "dropout": 0.1}, 0.4),
                ("glm", "glm", {"seeds": 8, "glm_l2": 1e-3, "glm_dk": 1e-1}, 0.6)]
CLI_OVER: dict = {}
WINDOW_KEYS = {"jitter", "dec31", "generic", "generic_step", "extra_levels", "pool", "base_long", "base_short",
               "store_analog", "sim_new"}


def write_submission(pred: np.ndarray, te: dict, out_dir: str) -> None:
    s = pd.Series(pred, index=te["rowid"])
    for part in ("public", "private"):
        ids = pd.read_csv(f"{DATA}/{part}_test/{part}_test.csv")["id"]
        sub = pd.DataFrame({"id": ids, "sales": np.clip(s.reindex(ids).to_numpy(), 0, None)})
        assert sub["sales"].notna().all()
        sub.to_csv(os.path.join(out_dir, f"{part}_submission.csv"), index=False)
        print("wrote", part, len(sub), "rows, total", round(sub.sales.sum()))


def finish(preds: list[np.ndarray], wts: list[float], te: dict) -> np.ndarray:
    """Blend (weighted geometric mean) + negative-binomial median + closed-day zeros."""
    pred = blend(preds, wts)
    if CFG["nb_q"] > 0:
        pred = nb_median(pred, CFG["nb_phi"], CFG["nb_q"])
    return np.where(te["lny"], 0.0, pred)


def fit_final_one(P: dict, kind: str, over: dict, base_cfg: dict, D: dict | None) -> tuple[np.ndarray, dict]:
    CFG.clear()
    CFG.update({**base_cfg, **over, **CLI_OVER})  # command-line overrides win (e.g. seed splits)
    if D is None or WINDOW_KEYS & set(over):
        D = prepare("FINAL", P)
    pred = postprocess(P, D["te"], fit_raw(D, kind), D["o"])
    CFG.clear()
    CFG.update(base_cfg)
    return pred, D


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="FINAL")
    ap.add_argument("--model", default="all", help="mlp | glm | hyb | all (final ensemble)")
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "oof"))
    ap.add_argument("--combine", action="store_true", help="blend saved FINAL_<name>.npz files")
    args = ap.parse_args()
    CLI_OVER.update(json.loads(args.cfg))
    CFG.update(CLI_OVER)
    torch.set_num_threads(CFG["threads"])
    if args.combine:  # parallel mode: each FINAL model was fitted by its own process
        zs = [np.load(os.path.join(args.out, f"FINAL_{name}.npz")) for name, _, _, _ in FINAL_MODELS]
        te = {"rowid": zs[0]["rowid"], "lny": zs[0]["lny"]}
        write_submission(finish([z["pred"] for z in zs], [w for *_, w in FINAL_MODELS], te), te, HERE)
        return
    P = build_panel(load())
    if args.fold != "FINAL":
        D = prepare(args.fold, P)
        raw = fit_raw(D, args.model)
        pred = postprocess(P, D["te"], raw, D["o"])
        print("backtest", args.fold, args.model, args.tag, report(P, D["te"], pred), flush=True)
        os.makedirs(args.out, exist_ok=True)
        te = D["te"]
        np.savez_compressed(os.path.join(args.out, f"{args.fold}_{args.model}_{args.tag}.npz"),
                            raw=raw, pred=pred, y=te["y"], w=te["w"], so=te["so"], si=te["si"], t=te["t"])
        return
    base_cfg = dict(CFG)
    print("I48 ratio:", round(i48_ratio(P, (pd.Timestamp(FOLDS["FINAL"][0]) - P["days"][0]).days), 3))
    if args.model != "all":  # parallel mode: fit one named member of FINAL_MODELS
        name, kind, over, _ = next(m for m in FINAL_MODELS if m[0] == args.model)
        pred, D = fit_final_one(P, kind, over, base_cfg, None)
        os.makedirs(args.out, exist_ok=True)
        np.savez_compressed(os.path.join(args.out, f"FINAL_{name}{args.tag}.npz"), pred=pred,
                            rowid=D["te"]["rowid"], lny=D["te"]["lny"])
        return
    D, preds = None, []
    for name, kind, over, _ in FINAL_MODELS:
        pred, D = fit_final_one(P, kind, over, base_cfg, D)
        preds.append(pred)
    write_submission(finish(preds, [w for *_, w in FINAL_MODELS], D["te"]), D["te"], HERE)


if __name__ == "__main__":
    main()
