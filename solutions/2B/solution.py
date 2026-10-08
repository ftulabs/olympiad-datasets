"""Warm-up 2B - rice-leaf disease, multi-modal (image + Vietnamese text + tabular).

End-to-end solution: per-modality experts, out-of-fold stacking.
  1. text expert : accent-stripped, de-noised text -> TF-IDF word(1-3) + char_wb(2-5) -> logistic regression
  2. tab expert  : numeric + categorical (soil/variety/season) + missing flag -> gradient boosting
  3. image expert: small BatchNorm CNN (and optionally pretrained ResNet-18) with flip/rot90/
                   colour/blur augmentation, OneCycle LR, label smoothing, 4-way TTA, 5-fold models
  4. fusion      : multinomial logistic regression on the experts' out-of-fold log-probabilities

Writes public_submission.csv / private_submission.csv (id,label) next to this file.
Usage:  python solution.py            (env: DATA_DIR, NT=threads, IMG_ARCHS="tiny,r18", EPOCHS_TINY, EPOCHS_R18)
"""
from __future__ import annotations

import os
import re
import time
import unicodedata

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

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", "/home/minh/Desktop/olympiad_ai/warmup/2B_rice_multimodal/dataset")
OUT_DIR = os.environ.get("OUT_DIR", HERE)
NT = int(os.environ.get("NT", "2"))
IMG_ARCHS = os.environ.get("IMG_ARCHS", "tiny").split(",")
EPOCHS = {"tiny": int(os.environ.get("EPOCHS_TINY", "25")), "r18": int(os.environ.get("EPOCHS_R18", "10"))}
N_FOLDS = 5
SEED = 0
torch.set_num_threads(NT)
DEV = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("USE_GPU", "1") == "1" else "cpu")

CLASSES = ["healthy", "blast", "brown_spot", "bacterial_blight", "nitrogen_deficiency"]
NUM = ["days_after_sowing", "humidity_7d", "temp_max_7d", "temp_min_7d", "rainfall_7d_mm",
       "storm_last_3d", "nitrogen_kg_ha", "field_area_ha"]
CAT = ["soil_type", "variety", "season"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- data
def load_split(split: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(f"{DATA_DIR}/{split}/{split}.csv")
    imgs = np.stack([np.asarray(Image.open(f"{DATA_DIR}/{split}/{p}").convert("RGB")) for p in df["image"]])
    return df, imgs  # images cached once as uint8 (N, 96, 96, 3)


def strip_accents(s: str) -> str:
    """Lower-case, remove Vietnamese diacritics, keep [a-z0-9], collapse repeated letters (quaaa -> qua)."""
    s = s.lower().replace("đ", "d")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"(.)\1+", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def make_text_cleaner(provinces: list[str]):
    """Province names are replaced by one token: test contains provinces never seen in train."""
    provs = sorted({strip_accents(p) for p in provinces}, key=len, reverse=True)

    def clean(s: str) -> str:
        s = strip_accents(s)
        for p in provs:
            s = s.replace(p, " tinhx ")
        return s
    return clean


def tab_matrix(df: pd.DataFrame, ref: pd.DataFrame) -> np.ndarray:
    X = df[NUM + CAT].copy()
    X["n_missing"] = df["nitrogen_kg_ha"].isna().astype(int)
    X["nitrogen_kg_ha"] = X["nitrogen_kg_ha"].fillna(-1)
    X["trange"] = df["temp_max_7d"] - df["temp_min_7d"]
    for c in CAT:
        X[c] = pd.Categorical(df[c], categories=sorted(ref[c].unique())).codes
    return X.values.astype(np.float64)


# ----------------------------------------------------------------------------- text / tab experts
def text_expert(t_tr: np.ndarray, y_tr: np.ndarray, t_te: np.ndarray) -> np.ndarray:
    vw = TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)
    A = sp.hstack([vw.fit_transform(t_tr), vc.fit_transform(t_tr)]).tocsr()
    B = sp.hstack([vw.transform(t_te), vc.transform(t_te)]).tocsr()
    return LogisticRegression(C=2, max_iter=3000).fit(A, y_tr).predict_proba(B)


CAT_MASK = [False] * len(NUM) + [True] * len(CAT) + [False, False]


def tab_expert(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
    m = HistGradientBoostingClassifier(learning_rate=0.02, max_iter=400, max_leaf_nodes=6, min_samples_leaf=30,
                                       l2_regularization=1.0, categorical_features=CAT_MASK, random_state=0)
    return m.fit(X_tr, y_tr).predict_proba(X_te)


# ----------------------------------------------------------------------------- image expert
def cbr(i: int, o: int, s: int = 1, k: int = 3) -> list[nn.Module]:
    return [nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


class TinyCNN(nn.Module):
    """BatchNorm VGG-style CNN for 96x96 leaves; avg+max pooled head."""

    def __init__(self, w: int = 24):
        super().__init__()
        self.f = nn.Sequential(*cbr(3, w, 2, 5), *cbr(w, w), nn.MaxPool2d(2),
                               *cbr(w, 2 * w), *cbr(2 * w, 2 * w), nn.MaxPool2d(2),
                               *cbr(2 * w, 4 * w), *cbr(4 * w, 4 * w), nn.MaxPool2d(2), *cbr(4 * w, 8 * w))
        self.h = nn.Sequential(nn.Dropout(0.2), nn.Linear(16 * w, 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.f(x)
        return self.h(torch.cat([Fn.adaptive_avg_pool2d(f, 1).flatten(1), Fn.adaptive_max_pool2d(f, 1).flatten(1)], 1))


def make_model(arch: str) -> nn.Module:
    if arch == "tiny":
        return TinyCNN(24)
    import torchvision
    m = torchvision.models.resnet18(weights="IMAGENET1K_V1")  # pretrained torchvision weights (allowed)
    m.fc = nn.Linear(512, 5)
    return m


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
BLUR_K = (lambda k: (k[:, None] * k[None]) / 256)(torch.tensor([1., 4., 6., 4., 1.]))


def prep(xb: torch.Tensor) -> torch.Tensor:
    return ((xb.to(DEV).float() / 255.0) - MEAN.to(DEV)) / STD.to(DEV)


def augment(xb: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """flips + rot90 (leaf orientation is arbitrary), brightness/colour-cast jitter, random Gaussian blur
    (test photos are on average blurrier than train)."""
    x = prep(xb)
    if rng.random() < 0.5:
        x = x.flip(3)
    if rng.random() < 0.5:
        x = x.flip(2)
    x = torch.rot90(x, int(rng.integers(4)), (2, 3))
    b = x.shape[0]
    x = x * (1 + 0.2 * (torch.rand(b, 1, 1, 1, device=DEV) - 0.5)) + 0.2 * (torch.rand(b, 3, 1, 1, device=DEV) - 0.5)
    m = torch.rand(b, device=DEV) < 0.3
    if m.any():
        w = BLUR_K.to(DEV).expand(3, 1, 5, 5).contiguous()
        xs = x[m]
        for _ in range(int(rng.integers(1, 4))):
            xs = Fn.conv2d(Fn.pad(xs, (2, 2, 2, 2), mode="reflect"), w, groups=3)
        x[m] = xs
    return x


@torch.no_grad()
def predict_img(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """4-way TTA: identity, h-flip, v-flip, rot90."""
    model.eval()
    out = []
    for i in range(0, len(X), 250):
        x = prep(X[i:i + 250])
        p = sum(model(t).softmax(1) for t in (x, x.flip(3), x.flip(2), torch.rot90(x, 1, (2, 3)))) / 4
        out.append(p.cpu())
    return torch.cat(out).numpy()


def train_img(arch: str, X: torch.Tensor, y: np.ndarray, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(arch).to(DEV)
    epochs, bs = EPOCHS[arch], 64
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * ((len(X) + bs - 1) // bs),
                                              pct_start=0.2)
    yt = torch.tensor(y, device=DEV)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            loss = Fn.cross_entropy(model(augment(X[idx], rng)), yt[idx.to(DEV)], label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
    return model


# ----------------------------------------------------------------------------- main
def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    return f1_score(y, p.argmax(1), average="macro")


def main() -> None:
    log(f"device={DEV} threads={NT} image archs={IMG_ARCHS} epochs={EPOCHS}")
    tr, img_tr = load_split("train")
    pu, img_pu = load_split("public_test")
    pr, img_pr = load_split("private_test")
    te = pd.concat([pu, pr], ignore_index=True)
    img_te = np.concatenate([img_pu, img_pr])
    y = tr["label"].map(CLASSES.index).values
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(tr, y))

    clean = make_text_cleaner(list(pd.concat([tr.province, te.province]).unique()))
    experts: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # name -> (oof probs, test probs)

    # text + tabular experts: OOF for the stacker, full-train fit for test
    T_tr, T_te = tr["text"].map(clean).values, te["text"].map(clean).values
    X_tr, X_te = tab_matrix(tr, tr), tab_matrix(te, tr)
    for name, fn, A, B in (("text", text_expert, T_tr, T_te), ("tab", tab_expert, X_tr, X_te)):
        oof = np.zeros((len(y), 5))
        for a, b in folds:
            oof[b] = fn(A[a], y[a], A[b])
        experts[name] = (oof, fn(A, y, B))
        log(f"{name:8s} OOF macro-F1={macro_f1(y, oof):.4f} logloss={log_loss(y, oof):.4f}")

    # image experts: 5 fold models -> OOF + averaged test prediction
    Xi_tr = torch.tensor(img_tr).permute(0, 3, 1, 2).contiguous()
    Xi_te = torch.tensor(img_te).permute(0, 3, 1, 2).contiguous()
    for arch in IMG_ARCHS:
        oof, tp = np.zeros((len(y), 5)), np.zeros((len(te), 5))
        for k, (a, b) in enumerate(folds):
            m = train_img(arch, Xi_tr[a], y[a], seed=SEED + k)
            oof[b] = predict_img(m, Xi_tr[b])
            tp += predict_img(m, Xi_te) / N_FOLDS
            log(f"img_{arch} fold{k} macro-F1={macro_f1(y[b], oof[b]):.4f}")
        experts[f"img_{arch}"] = (oof, tp)
        log(f"img_{arch} OOF macro-F1={macro_f1(y, oof):.4f}")

    # fusion: logistic regression on OOF log-probabilities (stacking)
    L = lambda p: np.log(np.clip(p, 1e-4, 1))  # noqa: E731
    Z_tr = np.hstack([L(o) for o, _ in experts.values()])
    Z_te = np.hstack([L(t) for _, t in experts.values()])
    stacker = lambda: LogisticRegression(C=0.3, max_iter=3000)  # noqa: E731
    cv_p = np.zeros((len(y), 5))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=123).split(Z_tr, y):
        cv_p[b] = stacker().fit(Z_tr[a], y[a]).predict_proba(Z_tr[b])
    log(f"STACK {'+'.join(experts)} CV macro-F1={macro_f1(y, cv_p):.4f}")
    p_te = stacker().fit(Z_tr, y).predict_proba(Z_te)

    labels = np.array(CLASSES)[p_te.argmax(1)]
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame({"id": pu["id"], "label": labels[:len(pu)]}).to_csv(f"{OUT_DIR}/public_submission.csv", index=False)
    pd.DataFrame({"id": pr["id"], "label": labels[len(pu):]}).to_csv(f"{OUT_DIR}/private_submission.csv", index=False)
    log(f"wrote submissions to {OUT_DIR}; predicted class mix {pd.Series(labels).value_counts(normalize=True).round(3).to_dict()}")


if __name__ == "__main__":
    main()
