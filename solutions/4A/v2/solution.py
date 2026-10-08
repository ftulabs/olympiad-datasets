"""4A - Community rules: comment moderation (Vietnamese) -- v2.

Pipeline
  1. normalise text (diacritics, teencode noise, cue tokens) -> TF-IDF char_wb 2-5 + word 1-2
  2. per-rule few-shot scorers from the example columns (LR, kNN margin, LR with other
     rules' positive examples as extra negatives)
  3. NEW: transductive "template density" features. Bodies posted under a rule are about
     half violations generated from that rule's templates, so a violation's near neighbours
     (over all 14k bodies of train+public+private) are enriched in the same rule, while
     allowed comments (chit-chat, other rules' violations, hard negatives) are spread over
     many rules. Features: share of same-rule bodies among the top-5/20/50 neighbours,
     number of distinct rules, near-duplicate enrichment, same-group density, neighbour
     similarity, neighbour-averaged few-shot score.
  4. rule-agnostic PyTorch stacker (trained from scratch) on rule-conditioned features. It is
     trained on simulated "unseen-rule" blocks: for each train rule, keep 250 or 550 of its
     bodies, drop the rest from the corpus and use a 40+40 example pool -- exactly the
     situation of the rules that only appear in the test files.
  5. rules seen in train: within-rule rank blend of a supervised per-rule LR (train rows +
     examples + other rules' positives as weak negatives) with the stacker.

Writes public_submission.csv / private_submission.csv next to this file.
Usage: python3 solution.py [--validate]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

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
from sklearn.preprocessing import normalize

torch.set_num_threads(2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA = Path(os.environ.get("DATA_4A", "/home/minh/Desktop/olympiad_ai/warmup/4A_comment_moderation/dataset"))
OUT = Path(globals().get("__file__", "solution.py")).resolve().parent
TARGET = "rule_violation"
C_LR = 10.0          # inverse L2 strength of every logistic regression
TOPK = 3             # neighbours of the kNN-margin few-shot score
CORR_SKIP = 0.5      # rules whose few-shot score columns correlate above this = "same topic"
W_POOLNEG = 0.3      # weight of other-topic rules' positive examples used as negatives
W_TRPOSNEG = 0.1     # weight of other-topic rules' positive train rows used as negatives (sup model)
KNN = 50             # neighbours for the density features
SIM_SIZES = (250, 550)  # body counts of the unseen rules in the test files (private only / pub+priv)
SIM_DRAWS = 3
SIM_POOL = 40        # unseen rules come with 40 positive + 40 negative examples
W_SUP = 0.4          # within-rule rank weight of the supervised model for seen rules
SEEDS = (0, 1, 2, 3, 4)
GROUPS = ("v1", "aug", "dens", "top", "ndup", "grp", "nbr")


def log(t0: float, msg: str) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


# ============================================================================ Data
def strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def normalise(text: str) -> str:
    """Lower-case, add tag tokens for strong cues, strip diacritics, squash teencode noise."""
    t = unicodedata.normalize("NFC", str(text)).lower()
    tags = []
    if re.match(r"^\s*\[\s*spoil", t):
        tags.append("zztagspoil")
    if "spoil" in t:
        tags.append("zzspoilword")
    if re.search(r"(\d[\s.\-]?){9,}", t):
        tags.append("zzphone")
    if re.search(r"\b[\w-]+\.(site|top|ly|id|xyz|cc|click|link|io|me)\b|bit\.ly|tinyurl|cutt|s\.id", t):
        tags.append("zzshortlink")
    if re.search(r"\b\d+\s?(k|nghìn|nghin|tr|củ|cu|đ|d)\b|\d+\.000", t):
        tags.append("zzprice")
    n_digits = len(re.sub(r"\D", "", t))
    t = strip_accents(t)
    t = re.sub(r"(.)\1{2,}", r"\1", t)
    t = re.sub(r"\d", "0", t)
    t = re.sub(r"[^\w\s\[\]./]", " ", t)
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
    """All distinct example comments per rule, pooled over every row of every split."""
    pools: dict[str, tuple[set, set]] = {}
    for d in dfs:
        for r, g in d.groupby("rule"):
            p, n = pools.setdefault(r, (set(), set()))
            p.update(g.positive_example_1, g.positive_example_2)
            n.update(g.negative_example_1, g.negative_example_2)
    return {r: (sorted(p), sorted(n)) for r, (p, n) in pools.items()}


def rank_in_rule(scores: np.ndarray, rules: np.ndarray) -> np.ndarray:
    return pd.Series(scores).groupby(np.asarray(rules)).rank(pct=True).values


def zz(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-9)


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


# =========================================================================== Model
def lr(X, y, w=None) -> LogisticRegression:
    return LogisticRegression(C=C_LR, max_iter=3000).fit(X, y, sample_weight=w)


class Context:
    """Everything shared by the feature builders: matrices, pools, few-shot score columns."""

    def __init__(self, tr: pd.DataFrame, pu: pd.DataFrame, pr: pd.DataFrame, t0: float):
        self.tr, self.pu, self.pr = tr, pu, pr
        self.pools = rule_pools([tr, pu, pr])
        self.rules = sorted(self.pools)
        self.allb = pd.concat([tr, pu, pr], ignore_index=True)
        self.ntr, self.npu = len(tr), len(pu)
        self.N = len(self.allb)
        self.ri = self.allb.rule.map({r: i for i, r in enumerate(self.rules)}).values
        self.grp = pd.factorize(self.allb.group)[0]
        self.y = tr[TARGET].values.astype(float)
        texts = sorted(set(self.allb.body) | {t for p, n in self.pools.values() for t in p + n})
        feat = Featuriser(texts)
        self.XA = feat(self.allb.body.tolist())
        self.Xn = normalize(self.XA)  # cosine geometry of the bodies
        self.XE = {r: (feat(p + n), np.r_[np.ones(len(p)), np.zeros(len(n))]) for r, (p, n) in self.pools.items()}
        log(t0, f"tf-idf {self.XA.shape}")
        R = len(self.rules)
        self.S1, self.K, self.Sa = (np.zeros((self.N, R)) for _ in range(3))
        # topic clusters from the plain few-shot LR columns
        for j in range(R):
            Xe, ye = self.XE[self.rules[j]]
            self.S1[:, j] = lr(Xe, ye).decision_function(self.XA)
        self.corr_topic = np.corrcoef(self.S1.T)
        for j in range(R):
            Xe, ye = self.XE[self.rules[j]]
            s1, k1, sa = self.fewshot_cols(j, np.where(ye == 1)[0], np.where(ye == 0)[0])
            self.K[:, j], self.Sa[:, j] = k1, sa
        self.corr_S1 = np.corrcoef(self.S1.T)
        self.corr_Sa = np.corrcoef(self.Sa.T)
        self.mu1, self.sd1 = self.S1.mean(0), self.S1.std(0) + 1e-9
        self.mua, self.sda = self.Sa.mean(0), self.Sa.std(0) + 1e-9
        log(t0, "few-shot score matrices")

    def other_pos(self, j: int):
        """Positive examples of every other-topic rule (used as weak negatives for rule j)."""
        Xs, n = [], 0
        for k in range(len(self.rules)):
            if k == j or self.corr_topic[j, k] >= CORR_SKIP:
                continue
            Xk, yk = self.XE[self.rules[k]]
            Xs.append(Xk[yk == 1])
        X = vstack(Xs)
        return X, np.zeros(X.shape[0]), np.full(X.shape[0], W_POOLNEG)

    def fewshot_cols(self, j: int, ip: np.ndarray, ineg: np.ndarray):
        """Few-shot LR, kNN margin and LR+other-positives for rule j from a given example pool."""
        Xe, _ = self.XE[self.rules[j]]
        Xs = vstack([Xe[ip], Xe[ineg]])
        ys = np.r_[np.ones(len(ip)), np.zeros(len(ineg))]
        s1 = lr(Xs, ys).decision_function(self.XA)
        sim = (self.XA @ Xs.T).toarray() / 2.0
        k1 = np.sort(sim[:, :len(ip)], 1)[:, -TOPK:].mean(1) - np.sort(sim[:, len(ip):], 1)[:, -TOPK:].mean(1)
        Xo, yo, wo = self.other_pos(j)
        sa = lr(vstack([Xs, Xo]), np.r_[ys, yo], np.r_[np.ones(len(ys)), wo]).decision_function(self.XA)
        return s1, k1, sa

    # ------------------------------------------------------------------ features
    def other_max(self, S, mu, sd, corr, rows, j):
        o = ((S[rows] - mu) / sd).copy()
        o[:, corr[j] > CORR_SKIP] = -9.0
        return o.max(1)

    def block_features(self, j: int, rows: np.ndarray, corpus: np.ndarray, cols=None) -> dict[str, np.ndarray]:
        """Rule-conditioned features for `rows` (all posted under rule j). `corpus` masks which
        bodies exist (used to simulate unseen rules); `cols` overrides rule j's few-shot columns."""
        S1, Sa = self.S1, self.Sa
        if cols is not None:
            S1, Sa = S1.copy(), Sa.copy()
            S1[:, j], k1, Sa[:, j] = cols[0], cols[1][rows], cols[2]
        else:
            k1 = self.K[rows, j]
        ri = self.ri
        sim = (self.Xn[rows] @ self.Xn.T).toarray()
        sim[:, ~corpus] = -2.0
        sim[np.arange(len(rows)), rows] = -2.0
        nn_ = np.argsort(-sim, 1)[:, :KNN]
        ns = np.take_along_axis(sim, nn_, 1)
        nr = ri[nn_]
        same = nr == j
        n_distinct = np.array([len(set(x)) for x in nr[:, :20]])
        ndup = []
        for thr in (0.5, 0.7):
            m = ns > thr
            cnt = m.sum(1)
            ndup += [(same & m).sum(1) / (cnt + 1.0), np.log1p(cnt)]
        gsim = np.where(self.grp[None, :] == self.grp[rows][:, None], sim, -2.0)
        hg = (ri[np.argsort(-gsim, 1)[:, :20]] == j).mean(1)
        del sim, gsim
        s1, sa = S1[rows, j], Sa[rows, j]
        o1 = self.other_max(S1, self.mu1, self.sd1, self.corr_S1, rows, j)
        oa = self.other_max(Sa, self.mua, self.sda, self.corr_Sa, rows, j)
        return {
            "v1": np.c_[zz(s1), zz(k1), o1, zz(s1) - o1],
            "aug": np.c_[zz(sa), zz(sa) - oa],
            "dens": np.c_[same[:, :5].mean(1), same[:, :20].mean(1), same.mean(1), n_distinct],
            "top": np.c_[ns[:, 0], ns[:, :5].mean(1)],
            "ndup": np.c_[tuple(ndup)],
            "grp": np.c_[hg],
            "nbr": np.c_[zz(Sa[nn_[:, :5], j].mean(1)), zz(Sa[nn_[:, :20], j].mean(1))],
        }

    def sim_blocks(self, seed: int = 0) -> list[dict]:
        """Training blocks for the stacker: each seen rule as a simulated unseen rule
        (250 / 550 bodies, rest of its bodies removed from the corpus, 40+40 pool) plus
        its natural full block."""
        rng = np.random.RandomState(seed)
        blocks = []
        for j in sorted(set(self.ri[:self.ntr])):
            allj = np.where(self.ri == j)[0]
            trj = np.where(self.ri[:self.ntr] == j)[0]
            _, ye = self.XE[self.rules[j]]
            ip, ineg = np.where(ye == 1)[0], np.where(ye == 0)[0]
            for n in SIM_SIZES:
                for _ in range(SIM_DRAWS):
                    keep = rng.choice(trj, n, replace=False)
                    corpus = np.ones(self.N, bool)
                    corpus[allj] = False
                    corpus[keep] = True
                    cols = self.fewshot_cols(j, rng.choice(ip, min(SIM_POOL, len(ip)), replace=False),
                                             rng.choice(ineg, min(SIM_POOL, len(ineg)), replace=False))
                    blocks.append(dict(j=j, n=n, rows=keep, F=self.block_features(j, keep, corpus, cols)))
            blocks.append(dict(j=j, n=len(allj), rows=trj, F=self.block_features(j, trj, np.ones(self.N, bool))))
        return blocks


class Stacker(nn.Module):
    """Small MLP over rule-conditioned features, trained from scratch, shared across rules."""

    def __init__(self, d_in: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(), nn.Dropout(0.1), nn.Linear(hidden, 1))
        self.skip = nn.Linear(d_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.net(x) + self.skip(x)).squeeze(-1)


def fit_stacker(F: np.ndarray, y: np.ndarray, Fpred: np.ndarray, epochs: int = 300) -> np.ndarray:
    mu, sd = F.mean(0), F.std(0) + 1e-9
    xt = torch.tensor((F - mu) / sd, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    xp = torch.tensor((Fpred - mu) / sd, dtype=torch.float32, device=DEVICE)
    out = np.zeros(len(Fpred))
    for seed in SEEDS:
        torch.manual_seed(seed)
        m = Stacker(F.shape[1]).to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=0.02, weight_decay=1e-3)
        lossf = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            m.train()
            opt.zero_grad()
            lossf(m(xt), yt).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            out += m(xp).cpu().numpy()
    return out / len(SEEDS)


def stackF(b: dict) -> np.ndarray:
    return np.hstack([b["F"][g] for g in GROUPS])


def supervised_rule_model(ctx: Context, j: int, rows_tr: np.ndarray, rows_pred: np.ndarray) -> np.ndarray:
    """Per-rule LR: the rule's labelled train rows + its examples + other-topic positives as
    weak negatives (examples and train rows of other-topic rules)."""
    Xe, ye = ctx.XE[ctx.rules[j]]
    Xo, yo, wo = ctx.other_pos(j)
    others = [k for k in range(len(ctx.rules)) if k != j and ctx.corr_topic[j, k] < CORR_SKIP]
    tp = np.where(np.isin(ctx.ri[:ctx.ntr], others) & (ctx.y == 1))[0]
    X = vstack([ctx.XA[rows_tr], Xe, Xo, ctx.XA[tp]])
    yy = np.r_[ctx.y[rows_tr], ye, yo, np.zeros(len(tp))]
    w = np.r_[np.ones(len(rows_tr) + len(ye)), wo, np.full(len(tp), W_TRPOSNEG)]
    return lr(X, yy, w).decision_function(ctx.XA[rows_pred])


# ============================================================ Training & inference
def validate(ctx: Context, blocks: list[dict], t0: float) -> None:
    """LORO on simulated blocks (= unseen-rule protocol) and 5-fold within rule (= seen)."""
    seen = sorted(set(ctx.ri[:ctx.ntr]))
    res: dict = {}
    full_oof = np.zeros(ctx.ntr)
    for j in seen:
        trb = [b for b in blocks if b["j"] != j]
        teb = [b for b in blocks if b["j"] == j]
        p = fit_stacker(np.vstack([stackF(b) for b in trb]), np.concatenate([ctx.y[b["rows"]] for b in trb]),
                        np.vstack([stackF(b) for b in teb]))
        o = 0
        for b in teb:
            pb = p[o:o + len(b["rows"])]
            o += len(b["rows"])
            key = b["n"] if b["n"] in SIM_SIZES else "full"
            res.setdefault(key, {}).setdefault(j, []).append(roc_auc_score(ctx.y[b["rows"]], pb))
            if key == "full":
                full_oof[b["rows"]] = pb
    for k, d in res.items():
        print(f"  VAL stacker LORO, rule size {k!s:5s}: {np.mean([np.mean(v) for v in d.values()]):.4f}", flush=True)
    sup = np.zeros(ctx.ntr)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for j in seen:
        m = np.where(ctx.ri[:ctx.ntr] == j)[0]
        for a, b in skf.split(m, ctx.y[m]):
            sup[m[b]] = supervised_rule_model(ctx, j, m[a], m[b])
    rr = ctx.ri[:ctx.ntr]

    def mauc(p):
        return np.mean([roc_auc_score(ctx.y[rr == j], p[rr == j]) for j in seen])

    print(f"  VAL supervised per-rule 5-fold: {mauc(sup):.4f}", flush=True)
    for w in (0.2, 0.3, 0.4, 0.5, 0.6):
        print(f"  VAL seen blend w_sup={w}: {mauc(w * rank_in_rule(sup, rr) + (1 - w) * rank_in_rule(full_oof, rr)):.4f}",
              flush=True)
    log(t0, "validation done")


def main(do_validate: bool) -> None:
    t0 = time.time()
    tr, pu, pr = load()
    ctx = Context(tr, pu, pr, t0)
    blocks = ctx.sim_blocks()
    log(t0, f"{len(blocks)} simulated training blocks")
    if do_validate:
        validate(ctx, blocks, t0)

    # stacker trained on every block of every seen rule, applied to every test body
    Ftr = np.vstack([stackF(b) for b in blocks])
    ytr = np.concatenate([ctx.y[b["rows"]] for b in blocks])
    final = np.full(ctx.N, np.nan)
    seen = set(ctx.ri[:ctx.ntr])
    full = np.ones(ctx.N, bool)
    test_idx = np.arange(ctx.ntr, ctx.N)
    pred_rows, pred_F = [], []
    for j in range(len(ctx.rules)):
        rows = test_idx[ctx.ri[ctx.ntr:] == j]
        pred_rows.append(rows)
        pred_F.append(stackF({"F": ctx.block_features(j, rows, full)}))
    meta = np.full(ctx.N, np.nan)
    meta[np.concatenate(pred_rows)] = fit_stacker(Ftr, ytr, np.vstack(pred_F))
    log(t0, "stacker predictions")
    rules_col = ctx.allb.rule.values
    for j in range(len(ctx.rules)):
        rows = test_idx[ctx.ri[ctx.ntr:] == j]
        r_meta = pd.Series(meta[rows]).rank(pct=True).values
        if j in seen:
            trj = np.where(ctx.ri[:ctx.ntr] == j)[0]
            r_sup = pd.Series(supervised_rule_model(ctx, j, trj, rows)).rank(pct=True).values
            final[rows] = W_SUP * r_sup + (1 - W_SUP) * r_meta
        else:
            final[rows] = r_meta
    log(t0, "supervised models + blend")

    sub_pu = pd.DataFrame({"id": pu["id"].values, TARGET: final[ctx.ntr:ctx.ntr + ctx.npu]})
    sub_pr = pd.DataFrame({"id": pr["id"].values, TARGET: final[ctx.ntr + ctx.npu:]})
    for sub, df, name in ((sub_pu, pu, "public_submission.csv"), (sub_pr, pr, "private_submission.csv")):
        assert len(sub) == len(df) and sub[TARGET].notna().all()
        sub.to_csv(OUT / name, index=False)
    log(t0, f"wrote {OUT / 'public_submission.csv'} and private_submission.csv")


if __name__ == "__main__":
    main(do_validate="--validate" in sys.argv)
