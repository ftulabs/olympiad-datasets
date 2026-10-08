"""4A - Community rules: comment moderation (Vietnamese).

End-to-end solution: normalisation -> TF-IDF (char + word n-grams) -> per-rule
few-shot classifiers built from the example columns -> rule-conditioned PyTorch
stacker (trained from scratch) -> per-rule rank blend with a supervised per-rule
model for rules that appear in train.

Writes public_submission.csv / private_submission.csv next to this file.
Usage: python3 solution.py [--validate]
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"

import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

torch.set_num_threads(1)

DATA = Path("/home/minh/Desktop/olympiad_ai/warmup/4A_comment_moderation/dataset")
OUT = Path(__file__).resolve().parent
TARGET = "rule_violation"
C_LR = 10.0  # inverse L2 strength of every logistic regression
TOPK = 3  # neighbours used by the kNN similarity feature
CORR_SKIP = 0.5  # rules whose score columns correlate above this are "the same topic"
BLEND_W_SUP = 0.75  # weight of the supervised per-rule model for rules seen in train
SEEDS = (0, 1, 2, 3, 4)


# ----------------------------------------------------------------------------- data
def strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def normalise(text: str) -> str:
    """Lower-case, add tag tokens for strong cues, strip diacritics, squash teencode noise."""
    t = unicodedata.normalize("NFC", str(text)).lower()
    tags = []
    if re.match(r"^\s*\[\s*spoil", t):  # [SPOIL] written at the very start
        tags.append("zztagspoil")
    if "spoil" in t:
        tags.append("zzspoilword")
    if re.search(r"(\d[\s.\-]?){9,}", t):  # phone / CCCD numbers, also "0637.676.809"
        tags.append("zzphone")
    if re.search(r"\b[\w-]+\.(site|top|ly|id|xyz|cc|click|link|io|me)\b|bit\.ly|tinyurl|cutt|s\.id", t):
        tags.append("zzshortlink")
    if re.search(r"\b\d+\s?(k|nghìn|nghin|tr|củ|cu|đ|d)\b|\d+\.000", t):
        tags.append("zzprice")
    n_digits = len(re.sub(r"\D", "", t))
    t = strip_accents(t)
    t = re.sub(r"(.)\1{2,}", r"\1", t)  # "chỉỉỉỉ" -> "chi"
    t = re.sub(r"\d", "0", t)  # every digit looks the same
    t = re.sub(r"[^\w\s\[\]./]", " ", t)  # emoji / punctuation out
    t = re.sub(r"\s+", " ", t).strip()
    if n_digits:
        tags.append(f"zzndig{min(n_digits, 12)}")
    return t + " " + " ".join(tags)


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(DATA / "train/train.csv")
    pu = pd.read_csv(DATA / "public_test/public_test.csv")
    pr = pd.read_csv(DATA / "private_test/private_test.csv")
    return tr, pu, pr


def rule_pools(dfs: list[pd.DataFrame]) -> dict[str, tuple[list[str], list[str]]]:
    """All distinct labelled example comments per rule, collected over every split."""
    pools: dict[str, tuple[set, set]] = {}
    for d in dfs:
        for r, g in d.groupby("rule"):
            p, n = pools.setdefault(r, (set(), set()))
            p.update(g.positive_example_1, g.positive_example_2)
            n.update(g.negative_example_1, g.negative_example_2)
    return {r: (sorted(p), sorted(n)) for r, (p, n) in pools.items()}


def col_auc(rules, y, p) -> tuple[float, dict]:
    d = pd.DataFrame({"r": np.asarray(rules), "y": np.asarray(y), "p": np.asarray(p)})
    res = {r: roc_auc_score(g.y, g.p) for r, g in d.groupby("r") if g.y.nunique() == 2}
    return float(np.mean(list(res.values()))), res


def rank_in_rule(scores: np.ndarray, rules: np.ndarray) -> np.ndarray:
    return pd.Series(scores).groupby(np.asarray(rules)).rank(pct=True).values


def zscore_in_rule(scores: np.ndarray, rules: np.ndarray) -> np.ndarray:
    s = pd.Series(scores)
    g = s.groupby(np.asarray(rules))
    return ((s - g.transform("mean")) / (g.transform("std") + 1e-9)).values


# ---------------------------------------------------------------------------- model
class Featuriser:
    """TF-IDF char_wb 2-5 + word 1-2 on normalised text, fitted on all (unlabelled) text."""

    def __init__(self, texts: list[str]):
        self.cache = {t: normalise(t) for t in texts}
        corpus = list(self.cache.values())
        self.vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                  sublinear_tf=True, max_features=200000).fit(corpus)
        self.vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                                  token_pattern=r"\S+").fit(corpus)

    def __call__(self, texts: list[str]) -> csr_matrix:
        c = [self.cache.get(t) or normalise(t) for t in texts]
        return hstack([self.vc.transform(c), self.vw.transform(c)]).tocsr()


def lr(X, y, w=None) -> LogisticRegression:
    return LogisticRegression(C=C_LR, max_iter=3000).fit(X, y, sample_weight=w)


def few_shot_scores(feat: Featuriser, pools: dict, rules: list[str], XA: csr_matrix):
    """Score every body under every rule using only that rule's example comments.
    S: logistic-regression margin, K: top-k cosine to positives minus to negatives."""
    S = np.zeros((XA.shape[0], len(rules)))
    K = np.zeros_like(S)
    for j, r in enumerate(rules):
        p, n = pools[r]
        Xe = feat(p + n)
        ye = np.r_[np.ones(len(p)), np.zeros(len(n))]
        S[:, j] = lr(Xe, ye).decision_function(XA)
        sim = (XA @ Xe.T).toarray() / 2.0  # two L2-normalised blocks -> cosine in [0,1]
        K[:, j] = (np.sort(sim[:, : len(p)], 1)[:, -TOPK:].mean(1)
                   - np.sort(sim[:, len(p):], 1)[:, -TOPK:].mean(1))
    return S, K


def meta_features(S: np.ndarray, K: np.ndarray, rule_idx: np.ndarray, rules_col: np.ndarray) -> np.ndarray:
    """Rule-conditioned features: own-rule few-shot scores + 'is it rather another topic?'."""
    idx = np.arange(len(rule_idx))
    corr = np.corrcoef(S.T)
    Sz = (S - S.mean(0)) / (S.std(0) + 1e-9)
    other = Sz.copy()
    for a in np.unique(rule_idx):
        rows = idx[rule_idx == a]
        same_topic = (corr[a] > CORR_SKIP)  # includes the rule itself
        other[np.ix_(rows, np.where(same_topic)[0])] = -9.0
    own, ownk = S[idx, rule_idx], K[idx, rule_idx]
    oth = other.max(1)
    return np.c_[zscore_in_rule(own, rules_col), zscore_in_rule(ownk, rules_col),
                 oth, zscore_in_rule(own, rules_col) - oth]


class Stacker(nn.Module):
    """Small MLP over rule-conditioned features, trained from scratch (shared across rules)."""

    def __init__(self, d_in: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(), nn.Dropout(0.1), nn.Linear(hidden, 1))
        self.skip = nn.Linear(d_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.net(x) + self.skip(x)).squeeze(-1)


def fit_stacker(F: np.ndarray, y: np.ndarray, Fpred: np.ndarray, epochs: int = 300) -> np.ndarray:
    mu, sd = F.mean(0), F.std(0) + 1e-9
    xt = torch.tensor((F - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    xp = torch.tensor((Fpred - mu) / sd, dtype=torch.float32)
    out = np.zeros(len(Fpred))
    for seed in SEEDS:
        torch.manual_seed(seed)
        m = Stacker(F.shape[1])
        opt = torch.optim.AdamW(m.parameters(), lr=0.02, weight_decay=1e-3)
        lossf = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            m.train()
            opt.zero_grad()
            lossf(m(xt), yt).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            out += m(xp).numpy()
    return out / len(SEEDS)


def supervised_rule_model(Xb_tr, yb, Xe, ye, Xpred) -> np.ndarray:
    """Per-rule LR on the labelled train rows of that rule plus its example comments."""
    return lr(vstack([Xb_tr, Xe]), np.r_[yb, ye]).decision_function(Xpred)


# ------------------------------------------------------------------------- pipeline
def main(validate: bool) -> None:
    t0 = time.time()
    tr, pu, pr = load()
    pools = rule_pools([tr, pu, pr])
    RULES = sorted(pools)
    allb = pd.concat([tr, pu, pr], ignore_index=True)
    ntr, npu = len(tr), len(pu)
    texts = sorted(set(allb.body) | {t for p, n in pools.values() for t in p + n})
    feat = Featuriser(texts)
    XA = feat(allb.body.tolist())
    rule_idx = allb.rule.map({r: i for i, r in enumerate(RULES)}).values
    rules_col = allb.rule.values
    print(f"[{time.time()-t0:5.1f}s] tf-idf {XA.shape}", flush=True)

    S, K = few_shot_scores(feat, pools, RULES, XA)
    F = meta_features(S, K, rule_idx, rules_col)
    y = tr[TARGET].values.astype(float)
    print(f"[{time.time()-t0:5.1f}s] few-shot score matrix {S.shape}", flush=True)

    seen = set(tr.rule)
    sup = np.full(len(allb), np.nan)
    if validate:
        # leave-one-rule-out stacker = how an unseen rule is scored
        meta_oof = np.zeros(ntr)
        for r in seen:
            m = (tr.rule == r).values
            meta_oof[m] = fit_stacker(F[:ntr][~m], y[~m], F[:ntr][m])
        # 5-fold supervised per-rule model = how a seen rule is scored
        sup_oof = np.zeros(ntr)
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        for r, g in tr.groupby("rule"):
            p, n = pools[r]
            Xe, ye = feat(p + n), np.r_[np.ones(len(p)), np.zeros(len(n))]
            Xb, yb = XA[g.index], g[TARGET].values
            for a, b in skf.split(Xb, yb):
                sup_oof[g.index[b]] = supervised_rule_model(Xb[a], yb[a], Xe, ye, Xb[b])
        rr = tr.rule.values
        rep = {
            "few-shot LR only (own S)": S[np.arange(ntr), rule_idx[:ntr]],
            "few-shot kNN only (own K)": K[np.arange(ntr), rule_idx[:ntr]],
            "stacker LORO (unseen-rule sim.)": meta_oof,
            "supervised per-rule 5-fold": sup_oof,
        }
        for w in (0.4, 0.5, 0.6, 0.7, 0.8):
            rep[f"blend w_sup={w}"] = w * rank_in_rule(sup_oof, rr) + (1 - w) * rank_in_rule(meta_oof, rr)
        for k, v in rep.items():
            print(f"  VAL {k:34s} {col_auc(rr, y, v)[0]:.4f}", flush=True)

    # final fit: stacker on all train rows, supervised model per seen rule on all its rows
    meta = fit_stacker(F[:ntr], y, F)
    for r in seen:
        p, n = pools[r]
        Xe, ye = feat(p + n), np.r_[np.ones(len(p)), np.zeros(len(n))]
        mtr = (tr.rule == r).values
        mte = np.r_[np.zeros(ntr, bool), (allb.rule.values[ntr:] == r)]
        sup[mte] = supervised_rule_model(XA[:ntr][mtr], y[mtr], Xe, ye, XA[mte])
    final = rank_in_rule(meta, rules_col)
    has_sup = ~np.isnan(sup)
    sup_rank = rank_in_rule(np.where(has_sup, sup, 0.0), rules_col)
    final = np.where(has_sup, BLEND_W_SUP * sup_rank + (1 - BLEND_W_SUP) * final, final)

    sub_pu = pd.DataFrame({"id": pu["id"].values, TARGET: final[ntr:ntr + npu]})
    sub_pr = pd.DataFrame({"id": pr["id"].values, TARGET: final[ntr + npu:]})
    for sub, df, name in ((sub_pu, pu, "public_submission.csv"), (sub_pr, pr, "private_submission.csv")):
        assert len(sub) == len(df) and sub[TARGET].notna().all()
        sub.to_csv(OUT / name, index=False)
    print(f"[{time.time()-t0:5.1f}s] wrote {OUT/'public_submission.csv'} and private_submission.csv", flush=True)


if __name__ == "__main__":
    main(validate="--validate" in sys.argv)
