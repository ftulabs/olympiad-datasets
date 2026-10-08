"""Warm-up 2B v2 - rice-leaf disease, multi-modal (image + Vietnamese text + tabular).

End-to-end pipeline (5-fold CV everywhere, same folds as v1):
  1. text expert : accent-stripped, typo-normalised, province-masked text -> TF-IDF word 1-3 + char_wb 2-5 -> LR
  2. tab expert  : numeric + categorical + engineered features -> HistGradientBoosting (shallow, slow LR)
  3. image experts: ImageNet-pretrained ResNet-18 (torchvision) at 96px and at 112px (2 seeds), 30 epochs,
     AdamW + OneCycle, dihedral + colour + Gaussian-blur augmentation (test has 2x more blurred photos),
     label smoothing, 8-way dihedral TTA, 5 fold models averaged on test
  4. (optional, off) fold-clean pseudo-labelling of the test photos (set pl=True in IMG_EXPERTS): helps a
     from-scratch CNN but hurt the pretrained ResNet stack, so it is not used in the final ensemble
  5. fusion: multinomial LR on the experts' out-of-fold log-probabilities; text/tab refit on all train for test,
     image test predictions = mean of the 5 fold models.
Writes public_submission.csv / private_submission.csv (id,label) to OUT_DIR (default: this folder).

usage: python solution.py      env: DATA_DIR, OUT_DIR, NT (threads), CACHE_DIR (intermediate preds, resumable),
                                    QUICK=1 (1-epoch smoke test)
"""
from __future__ import annotations

import os
import re
import sys
import time
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
D = os.environ.get("DATA_DIR", "/home/minh/Desktop/olympiad_ai/warmup/2B_rice_multimodal/dataset")
OUT_DIR = os.environ.get("OUT_DIR", HERE)
CACHE = os.environ.get("CACHE_DIR", os.path.join(HERE, "cache"))
QUICK = os.environ.get("QUICK", "0") == "1"
torch.set_num_threads(int(os.environ.get("NT", "4")))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cuda":
    torch.backends.cudnn.benchmark = True
    if os.environ.get("VRAM_FRAC"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ["VRAM_FRAC"]), 0)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


CLASSES=["healthy","blast","brown_spot","bacterial_blight","nitrogen_deficiency"]
NUM=["days_after_sowing","humidity_7d","temp_max_7d","temp_min_7d","rainfall_7d_mm","storm_last_3d","nitrogen_kg_ha","field_area_ha"]
CAT=["soil_type","variety","season"]
def strip_accents(s):
    s=s.lower().replace('đ','d'); s=unicodedata.normalize('NFKD',s); s=''.join(c for c in s if not unicodedata.combining(c))
    s=re.sub(r'[^a-z0-9 ]+',' ',s); s=re.sub(r'(.)\1+',r'\1',s); return re.sub(r'\s+',' ',s).strip()
def make_cleaner(provinces):
    provs=sorted({strip_accents(p) for p in provinces},key=len,reverse=True)
    def clean(s):
        s=' '+strip_accents(s)+' '
        for p in provs: s=s.replace(' '+p+' ',' tinhx ')
        s=re.sub(r'\b\d+\b',' numx ',s)
        return re.sub(r'\s+',' ',s).strip()
    return clean


def typo_fixer(texts):
    cnt=Counter(w for t in texts for w in t.split()); good=[w for w,c in cnt.items() if c>=10]
    def ed1(a,b):
        if abs(len(a)-len(b))>1: return False
        if len(a)==len(b):
            d=[i for i in range(len(a)) if a[i]!=b[i]]
            return len(d)==1 or (len(d)==2 and d[1]==d[0]+1 and a[d[0]]==b[d[1]] and a[d[1]]==b[d[0]])
        if len(a)>len(b): a,b=b,a
        return any(b[:i]+b[i+1:]==a for i in range(len(b)))
    fix={}
    for w,c in cnt.items():
        if c<10 and len(w)>=2:
            cands=[g for g in good if ed1(w,g)]
            if cands: fix[w]=max(cands,key=lambda g:cnt[g])
    return lambda t:' '.join(fix.get(w,w) for w in t.split())

def text_fit_predict(A,y,B,C=2.0):
    vw=TfidfVectorizer(ngram_range=(1,3),min_df=2,sublinear_tf=True); vc=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=3,sublinear_tf=True)
    XA=sp.hstack([vw.fit_transform(A),vc.fit_transform(A)]).tocsr(); XB=sp.hstack([vw.transform(B),vc.transform(B)]).tocsr()
    return LogisticRegression(C=C,max_iter=3000).fit(XA,y).predict_proba(XB)

def tab_matrix(df,ref):
    X=df[NUM+CAT].copy(); X['n_missing']=df.nitrogen_kg_ha.isna().astype(int); X['nitrogen_kg_ha']=X.nitrogen_kg_ha.fillna(-1)
    X['trange']=df.temp_max_7d-df.temp_min_7d
    for c in CAT: X[c]=pd.Categorical(df[c],categories=sorted(ref[c].unique())).codes
    X['n_per_day']=df.nitrogen_kg_ha.fillna(-1)/(df.days_after_sowing+1); X['rain_storm']=df.rainfall_7d_mm*(1+df.storm_last_3d); X['hum_tmin']=df.humidity_7d-3*df.temp_min_7d
    return X.values.astype(float)
CM=[False]*8+[True]*3+[False]*5
def tab_fit_predict(A,y,B):
    return HistGradientBoostingClassifier(learning_rate=0.01,max_iter=800,max_leaf_nodes=4,min_samples_leaf=30,l2_regularization=1.0,categorical_features=CM,random_state=0).fit(A,y).predict_proba(B)


# ------------------------------------------------------------------ models
def cbr(i, o, s=1, k=3):
    return [nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


class TinyCNN(nn.Module):
    def __init__(self, w=24, drop=0.2):
        super().__init__()
        self.f = nn.Sequential(*cbr(3, w, 2, 5), *cbr(w, w), nn.MaxPool2d(2),
                               *cbr(w, 2 * w), *cbr(2 * w, 2 * w), nn.MaxPool2d(2),
                               *cbr(2 * w, 4 * w), *cbr(4 * w, 4 * w), nn.MaxPool2d(2), *cbr(4 * w, 8 * w))
        self.h = nn.Sequential(nn.Dropout(drop), nn.Linear(16 * w, 5))

    def forward(self, x):
        f = self.f(x)
        return self.h(torch.cat([Fn.adaptive_avg_pool2d(f, 1).flatten(1), Fn.adaptive_max_pool2d(f, 1).flatten(1)], 1))


class GemHead(nn.Module):
    def __init__(self, c, drop=0.2):
        super().__init__()
        self.h = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * c, 5))

    def forward(self, f):
        return self.h(torch.cat([Fn.adaptive_avg_pool2d(f, 1).flatten(1), Fn.adaptive_max_pool2d(f, 1).flatten(1)], 1))


class TVNet(nn.Module):
    """torchvision backbone (pretrained) + avg/max head."""

    def __init__(self, arch: str):
        super().__init__()
        import torchvision.models as tvm
        nomp = arch.endswith("nm")
        base = arch[:-2] if nomp else arch
        if base in ("r18", "r34", "r50"):
            name = {"r18": "resnet18", "r34": "resnet34", "r50": "resnet50"}[base]
            m = getattr(tvm, name)(weights="DEFAULT")
            layers = [m.conv1, m.bn1, m.relu] + ([] if nomp else [m.maxpool]) + [m.layer1, m.layer2, m.layer3, m.layer4]
            self.body = nn.Sequential(*layers)
            c = m.fc.in_features
        elif base == "effb0":
            m = tvm.efficientnet_b0(weights="DEFAULT")
            self.body = m.features
            c = 1280
        elif base == "mnv3":
            m = tvm.mobilenet_v3_large(weights="DEFAULT")
            self.body = m.features
            c = 960
        elif base == "rgy8":
            m = tvm.regnet_y_800mf(weights="DEFAULT")
            self.body = nn.Sequential(m.stem, m.trunk_output)
            c = 784
        elif base == "cnxt":
            m = tvm.convnext_tiny(weights="DEFAULT")
            self.body = m.features
            c = 768
        elif base == "dn121":
            m = tvm.densenet121(weights="DEFAULT")
            self.body = nn.Sequential(m.features, nn.ReLU(inplace=True))
            c = 1024
        else:
            raise ValueError(arch)
        self.head = GemHead(c)

    def forward(self, x):
        return self.head(self.body(x))


def make_model(arch: str) -> nn.Module:
    if arch.startswith("tiny"):
        w = int(arch[4:] or 24)
        return TinyCNN(w)
    return TVNet(arch)


# ------------------------------------------------------------------ aug
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def gauss_kernel(sigma: float) -> torch.Tensor:
    r = max(1, int(3 * sigma + 0.5))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = gauss_kernel(sigma).to(x.device)
    r = (len(k) - 1) // 2
    c = x.shape[1]
    x = Fn.conv2d(Fn.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(c, 1, 1, -1).contiguous(), groups=c)
    x = Fn.conv2d(Fn.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(c, 1, -1, 1).contiguous(), groups=c)
    return x


class Prep:
    def __init__(self, res: int):
        self.res = res
        self.mean, self.std = MEAN.to(DEV), STD.to(DEV)

    def __call__(self, xb: torch.Tensor) -> torch.Tensor:
        x = xb.to(DEV, non_blocking=True).float().div_(255.0)
        return x

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        if self.res != 96:
            x = Fn.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False)
        return x


def augment(x: torch.Tensor, rng: np.random.Generator, cfg: dict) -> torch.Tensor:
    """x in [0,1] on DEV. dihedral, colour jitter, random blur (match test blur: sigma ~1.0-1.6), noise, crop-shift."""
    b = x.shape[0]
    if rng.random() < 0.5:
        x = x.flip(3)
    if rng.random() < 0.5:
        x = x.flip(2)
    x = torch.rot90(x, int(rng.integers(4)), (2, 3))
    # per-sample brightness / colour cast
    x = x * (1 + cfg["bj"] * (torch.rand(b, 1, 1, 1, device=DEV) - 0.5)) + cfg["cj"] * (torch.rand(b, 3, 1, 1, device=DEV) - 0.5)
    # random translation (wrap-free) via roll + small scale
    if cfg["shift"] > 0:
        s = int(rng.integers(-cfg["shift"], cfg["shift"] + 1)), int(rng.integers(-cfg["shift"], cfg["shift"] + 1))
        x = torch.roll(x, s, (2, 3))
    m = torch.rand(b, device=DEV) < cfg["blur_p"]
    if m.any():
        sig = float(rng.uniform(cfg["blur_lo"], cfg["blur_hi"]))
        x[m] = blur(x[m], sig)
    return x


@torch.no_grad()
def predict(model, X: torch.Tensor, prep: Prep, sigma: float = 0.0, tta: int = 1 if QUICK else 8) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), 128):
        x = prep(X[i:i + 128])
        if sigma > 0:
            x = blur(x, sigma)
        x = prep.norm(x)
        views = [x, x.flip(3), x.flip(2), torch.rot90(x, 1, (2, 3)),
                 torch.rot90(x, 2, (2, 3)), torch.rot90(x, 3, (2, 3)), torch.rot90(x.flip(3), 1, (2, 3)),
                 torch.rot90(x.flip(2), 1, (2, 3))][:tta]
        p = sum(model(v).float().softmax(1) for v in views) / len(views)
        out.append(p.cpu())
    return torch.cat(out).numpy()


def soft_ce(out, t):
    return -(t * Fn.log_softmax(out, 1)).sum(1).mean()


def train(arch, X, y, seed, cfg, Xv=None, yv=None, prep=None):
    """y: int labels (N,) or soft targets (N,5)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(arch).to(DEV)
    ep, bs = cfg["epochs"], cfg["bs"]
    pre = arch.startswith("tiny") is False
    params = model.parameters()
    if pre:
        bb = [p for n, p in model.named_parameters() if not n.startswith("head")]
        hd = [p for n, p in model.named_parameters() if n.startswith("head")]
        params = [{"params": bb, "lr": cfg["lr"]}, {"params": hd, "lr": cfg["lr"] * 5}]
        maxlr = [cfg["lr"], cfg["lr"] * 5]
    else:
        maxlr = cfg["lr"]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    steps = ep * (len(X) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=maxlr, total_steps=steps, pct_start=0.15)
    if y.ndim == 1:
        yt = Fn.one_hot(torch.tensor(y), 5).float()
    else:
        yt = torch.tensor(y).float()
    yt = (yt * (1 - cfg["ls"]) + cfg["ls"] / 5).to(DEV)
    for e in range(ep):
        model.train()
        perm = torch.randperm(len(X))
        t0, tl = time.time(), 0.0
        for i in range(0, len(X) - bs + 1, bs):
            idx = perm[i:i + bs]
            x = prep.norm(augment(prep(X[idx]), rng, cfg))
            yb = yt[idx.to(DEV)]
            if cfg["mix"] > 0 and rng.random() < cfg["mix"]:
                lam = float(rng.beta(1.0, 1.0))
                j = torch.randperm(len(idx), device=DEV)
                H = x.shape[2]
                cut = int(H * np.sqrt(1 - lam))
                cy, cx = int(rng.integers(H)), int(rng.integers(H))
                y1, y2, x1, x2 = max(cy - cut // 2, 0), min(cy + cut // 2, H), max(cx - cut // 2, 0), min(cx + cut // 2, H)
                x[:, :, y1:y2, x1:x2] = x[j][:, :, y1:y2, x1:x2]
                lam = 1 - (y2 - y1) * (x2 - x1) / (H * H)
                out = model(x)
                loss = soft_ce(out, lam * yb + (1 - lam) * yb[j])
            else:
                loss = soft_ce(model(x), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
            tl += loss.item()
        msg = f"ep{e} loss={tl / (len(X) // bs):.3f} {time.time() - t0:.0f}s"
        if Xv is not None and (e % 5 == 4 or e == ep - 1):
            from sklearn.metrics import f1_score
            pv = predict(model, Xv, prep, tta=1)
            msg += f" va F1={f1_score(yv, pv.argmax(1), average='macro'):.4f}"
        log(msg)
    return model




# ============================================================================ pipeline
IMG_EXPERTS = [  # final ensemble (see README ablations)
    dict(name="r18_s1", arch="r18", epochs=30, seed=1, bs=32, res=96, pl=False),
    dict(name="r18r112_s3", arch="r18", epochs=30, seed=3, bs=32, res=112, pl=False),
    # pseudo-label variant (ablation, not in the final ensemble: hurt the stack, see README):
    # dict(name="r18pl_s2", arch="r18", epochs=20, seed=2, bs=32, res=96, pl=True),
]
BASE_CFG = dict(lr=5e-4, wd=1e-2, ls=0.05, bj=0.2, cj=0.2, shift=0, blur_p=0.3, blur_lo=0.8, blur_hi=1.8, mix=0.0)
STACK_C = 0.3
L = lambda p: np.log(np.clip(p, 1e-4, 1))  # noqa: E731


def load_data():
    tr = pd.read_csv(f"{D}/train/train.csv")
    pu = pd.read_csv(f"{D}/public_test/public_test.csv")
    pr = pd.read_csv(f"{D}/private_test/private_test.csv")
    te = pd.concat([pu, pr], ignore_index=True)
    img = lambda df, s: np.stack([np.asarray(Image.open(f"{D}/{s}/{q}").convert("RGB")) for q in df["image"]])  # noqa: E731
    Itr = img(tr, "train")
    Ite = np.concatenate([img(pu, "public_test"), img(pr, "private_test")])
    return tr, te, pu, pr, Itr, Ite


def cached(name: str, fn):
    p = f"{CACHE}/{name}.npz"
    if os.path.exists(p) and not QUICK:
        z = np.load(p)
        return {k: z[k] for k in z.files}
    out = fn()
    os.makedirs(CACHE, exist_ok=True)
    np.savez(p, **out)
    return out


def tt_experts(tr, te, y, folds):
    clean = make_cleaner(pd.concat([tr.province, te.province]).unique())
    T, Tt = tr.text.map(clean).values, te.text.map(clean).values
    fx = typo_fixer(np.concatenate([T, Tt]))
    T, Tt = np.array([fx(t) for t in T]), np.array([fx(t) for t in Tt])
    X, Xt = tab_matrix(tr, tr), tab_matrix(te, tr)
    res = {}
    for name, fn, A, B in (("text", text_fit_predict, T, Tt), ("tab", tab_fit_predict, X, Xt)):
        def run(fn=fn, A=A, B=B):
            oof = np.zeros((len(y), 5))
            fold_test = []
            for a, b in folds:
                oof[b] = fn(A[a], y[a], A[b])
                fold_test.append(fn(A[a], y[a], B))  # used for fold-clean pseudo-labels
            return dict(oof=oof, test=fn(A, y, B), fold_test=np.stack(fold_test))
        res[name] = cached(name, run)
        log(f"{name:10s} OOF macro-F1={f1_score(y, res[name]['oof'].argmax(1), average='macro'):.4f}")
    return res


def image_expert(spec, Xtr, Xte, y, folds, pl_fold_test=None):
    def run():
        cfg = dict(BASE_CFG, epochs=1 if QUICK else spec["epochs"], bs=spec["bs"], res=spec["res"])
        prep = Prep(spec["res"])
        oof, fold_test = np.zeros((len(y), 5)), []
        for k, (a, b) in enumerate(folds):
            if QUICK:  # smoke test: tiny subset
                a = a[:192]
            Xa, ya = Xtr[a], np.eye(5)[y[a]]
            if pl_fold_test is not None:  # soft pseudo-labels for all test images, fitted without fold-k labels
                keep = slice(0, 64) if QUICK else slice(None)
                Xa, ya = torch.cat([Xa, Xte[keep]]), np.concatenate([ya, pl_fold_test[k][keep]])
            m = train(spec["arch"], Xa, ya, spec["seed"] * 100 + k, cfg, Xtr[b], y[b], prep)
            oof[b] = predict(m, Xtr[b], prep)
            fold_test.append(predict(m, Xte, prep))
            log(f"{spec['name']} fold{k} macro-F1={f1_score(y[b], oof[b].argmax(1), average='macro'):.4f}")
            del m
        fold_test = np.stack(fold_test)
        return dict(oof=oof, test=fold_test.mean(0), fold_test=fold_test)
    r = cached(spec["name"], run)
    log(f"{spec['name']:10s} OOF macro-F1={f1_score(y, r['oof'].argmax(1), average='macro'):.4f}")
    return r


def fold_clean_pseudo_labels(experts: dict, y, folds) -> np.ndarray:
    """P[k] = stack fitted on OOF rows outside fold k, applied to test predictions of fold-k-free experts."""
    P = []
    for k, (a, b) in enumerate(folds):
        Z = np.hstack([L(e["oof"]) for e in experts.values()])
        Zt = np.hstack([L(e["fold_test"][k]) for e in experts.values()])
        P.append(LogisticRegression(C=STACK_C, max_iter=5000).fit(Z[a], y[a]).predict_proba(Zt))
    return np.stack(P)


def main() -> None:
    log(f"device={DEV} quick={QUICK}")
    tr, te, pu, pr, Itr, Ite = load_data()
    y = tr["label"].map(CLASSES.index).values
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(tr, y))
    experts = tt_experts(tr, te, y, folds)

    Xtr = torch.tensor(Itr).permute(0, 3, 1, 2).contiguous()
    Xte = torch.tensor(Ite).permute(0, 3, 1, 2).contiguous()
    for spec in [s for s in IMG_EXPERTS if not s["pl"]]:
        experts[spec["name"]] = image_expert(spec, Xtr, Xte, y, folds)
    pl_specs = [s for s in IMG_EXPERTS if s["pl"]]
    if pl_specs:
        P = fold_clean_pseudo_labels(dict(experts), y, folds)
        for spec in pl_specs:
            experts[spec["name"]] = image_expert(spec, Xtr, Xte, y, folds, P)

    # stacking: multinomial LR on OOF log-probs; CV estimate with a different split
    Z = np.hstack([L(e["oof"]) for e in experts.values()])
    Zt = np.hstack([L(e["test"]) for e in experts.values()])
    cv = np.zeros((len(y), 5))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=123).split(Z, y):
        cv[b] = LogisticRegression(C=STACK_C, max_iter=5000).fit(Z[a], y[a]).predict_proba(Z[b])
    log(f"STACK {'+'.join(experts)}: CV macro-F1={f1_score(y, cv.argmax(1), average='macro'):.4f} "
        f"logloss={log_loss(y, cv):.4f}")
    p = LogisticRegression(C=STACK_C, max_iter=5000).fit(Z, y).predict_proba(Zt)
    labels = np.array(CLASSES)[p.argmax(1)]
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame({"id": pu["id"], "label": labels[:len(pu)]}).to_csv(f"{OUT_DIR}/public_submission.csv", index=False)
    pd.DataFrame({"id": pr["id"], "label": labels[len(pu):]}).to_csv(f"{OUT_DIR}/private_submission.csv", index=False)
    log(f"wrote submissions to {OUT_DIR}; class mix {pd.Series(labels).value_counts(normalize=True).round(3).to_dict()}")


if __name__ == "__main__":
    main()
