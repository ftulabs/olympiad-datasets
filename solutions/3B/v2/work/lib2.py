"""v2 experiment library: grouped private-mimic CV, GINE GNN with per-part pooling, block-ID embedding w/ dropout,
DeepSets / additive heads. Feature cache = v1 featurizer output (feats.pkl)."""
import math, os, pickle, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score as aps

HERE = os.path.dirname(os.path.abspath(__file__))
T = ['bind_BRD4', 'bind_HSA', 'bind_sEH']
DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PRIV_MIX = np.array([1532, 1143, 1115, 1210]) / 5000.0  # private rows by number of unseen blocks


def load(path=None):
    F_ = pickle.load(open(path or os.path.join(HERE, 'feats.pkl'), 'rb'))
    y = np.load(os.path.join(HERE, 'y.npy')).astype(np.float32)
    return F_, y


def mean_ap(y, p, w=None):
    return float(np.mean([aps(y[:, k], p[:, k], sample_weight=w) for k in range(y.shape[1])]))


def make_splits(d, n_folds=6, rand_frac=0.12, seed=0, scaf_hold=True):
    """fold k: hold out block group k (1/n_folds of blocks) + (optionally) one non-dominant scaffold
    + a random 12% of the remaining molecules (stratum nnew=0). Returns (tri, vai, nnew, newscaf)."""
    rng = np.random.RandomState(seed)
    blocks = np.unique(d['blocks']); rng.shuffle(blocks)
    groups = np.array_split(blocks, n_folds)
    us, uc = np.unique(d['scaf'], return_counts=True)
    small = list(us[np.argsort(uc)][:-1])  # all but the dominant scaffold
    out = []
    for f in range(n_folds):
        nnew = np.isin(d['blocks'], groups[f]).sum(1)
        ns = np.zeros(len(nnew), bool)
        if scaf_hold and f < len(small):
            ns = d['scaf'] == small[f]
        clean = np.where((nnew == 0) & ~ns)[0]
        r = np.random.RandomState(seed + 100 + f).rand(len(clean)) < rand_frac
        tri = clean[~r]
        vai = np.sort(np.concatenate([np.where((nnew > 0) | ns)[0], clean[r]]))
        out.append((tri, vai, nnew[vai], ns[vai]))
    return out


def mix_weights(nnew):
    """sample weights so that val strata (by # unseen blocks) match the private test mix."""
    cnt = np.bincount(nnew, minlength=4).astype(float)
    return (PRIV_MIX / np.maximum(cnt, 1))[nnew] * len(nnew)


def eval_val(y, p, nnew):
    w = mix_weights(nnew)
    r = {'all': mean_ap(y, p), 'mix': mean_ap(y, p, w)}
    for s in range(4):
        m = nnew == s
        if m.sum() > 50:
            r[f's{s}'] = mean_ap(y[m], p[m])
    return r


class Batcher:
    def __init__(self, d, idv, y=None, cut=False):
        self.d = d; self.y = y
        H = np.concatenate([d['scaf'][:, None], d['blocks']], 1)
        pos = np.clip(np.searchsorted(idv, H), 0, len(idv) - 1)
        self.pid = np.where(idv[pos] == H, pos + 1, 0).astype(np.int64)
        e = d['e'].astype(np.int64)
        mol_e = np.repeat(np.arange(len(d['ne'])), d['ne'])
        gi = e[:, 0] + d['ao'][mol_e]; gj = e[:, 1] + d['ao'][mol_e]
        cross = d['part'][gi] != d['part'][gj]
        att = np.zeros(len(d['part']), np.float32); att[gi[cross]] = 1; att[gj[cross]] = 1
        self.x = np.concatenate([d['x'].astype(np.float32), np.eye(4, dtype=np.float32)[d['part']], att[:, None]], 1)
        self.keep = ~cross if cut else np.ones(len(e), bool)
        self.e = e

    @staticmethod
    def _ranges(starts, lens):
        tot = lens.sum(); off = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
        return np.arange(tot) + off

    def batch(self, ids):
        d = self.d
        na = d['na'][ids]; ne = d['ne'][ids]
        aidx = self._ranges(d['ao'][ids], na); eidx = self._ranges(d['eo'][ids], ne)
        off = np.repeat(np.concatenate([[0], np.cumsum(na)[:-1]]), ne)
        e = self.e[eidx]; k = self.keep[eidx]
        s = (e[:, 0] + off)[k]; t = (e[:, 1] + off)[k]; et = e[k, 2]
        bi = np.repeat(np.arange(len(ids)), na)
        b = dict(x=self.x[aidx], src=np.concatenate([s, t]), dst=np.concatenate([t, s]), et=np.concatenate([et, et]),
                 pi=bi * 4 + d['part'][aidx].astype(np.int64), pid=self.pid[ids].ravel())
        if self.y is not None:
            b['y'] = self.y[ids]
        b = {k_: torch.from_numpy(v).to(DEV, non_blocking=True) for k_, v in b.items()}
        b['n'] = len(ids)
        return b


def scatter_sum(h, idx, n):
    return torch.zeros(n, h.size(1), dtype=h.dtype, device=h.device).index_add_(0, idx, h)


def scatter_max(h, idx, n):
    out = torch.full((n, h.size(1)), -1e4, dtype=h.dtype, device=h.device)
    return out.scatter_reduce(0, idx.unsqueeze(1).expand_as(h), h, 'amax', include_self=True)


class GINE(nn.Module):
    def __init__(self, din, hid=128, layers=4, drop=0.1, jk=False):
        super().__init__()
        self.inp = nn.Linear(din, hid)
        self.eemb = nn.ModuleList([nn.Embedding(4, hid) for _ in range(layers)])
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(hid, 2 * hid), nn.BatchNorm1d(2 * hid), nn.ReLU(),
                                                 nn.Linear(2 * hid, hid)) for _ in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hid) for _ in range(layers)])
        self.eps = nn.Parameter(torch.zeros(layers))
        self.drop = drop; self.jk = jk

    def forward(self, b):
        h = self.inp(b['x']); hs = []
        for l in range(len(self.mlps)):
            msg = F.relu(h[b['src']] + self.eemb[l](b['et']))
            z = self.mlps[l]((1 + self.eps[l]) * h + scatter_sum(msg, b['dst'], h.size(0)))
            h = h + F.dropout(F.relu(self.bns[l](z)), self.drop, self.training)
            hs.append(h)
        return torch.stack(hs, 0).mean(0) if self.jk else h


class Net(nn.Module):
    """GINE -> per-part sum/max pooling (+ID embedding w/ dropout) -> phi per part ->
    head 'ds': rho([phi(scaf), sum phi(blocks)]) ; 'add': sum of per-part logits ; 'both': ds + add."""

    def __init__(self, din=32, hid=128, layers=4, drop=0.3, gdrop=0.1, n_ids=0, emb=64, id_drop=0.5, head='ds',
                 jk=False, pdrop=0.0):
        super().__init__()
        self.gnn = GINE(din, hid, layers, gdrop, jk)
        self.n_ids, self.id_drop, self.head, self.pdrop = n_ids, id_drop, head, pdrop
        dd = 2 * hid
        if n_ids > 0:
            self.id_emb = nn.Embedding(n_ids + 1, emb, padding_idx=0)
            nn.init.normal_(self.id_emb.weight, std=0.05)
            with torch.no_grad():
                self.id_emb.weight[0].zero_()
            dd += emb
        self.part_emb = nn.Embedding(4, dd)
        self.phi = nn.Sequential(nn.LayerNorm(dd), nn.Linear(dd, 2 * hid), nn.ReLU(), nn.Dropout(drop),
                                 nn.Linear(2 * hid, hid), nn.ReLU())
        self.rho = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * hid, hid), nn.ReLU(), nn.Linear(hid, 3))
        self.addh = nn.Linear(hid, 3)

    def forward(self, b):
        n = b['n']
        h = self.gnn(b)
        z = torch.cat([scatter_sum(h, b['pi'], 4 * n), scatter_max(h, b['pi'], 4 * n).clamp(min=-50)], 1)
        if self.n_ids > 0:
            pid = b['pid']
            if self.training and self.id_drop > 0:
                pid = torch.where(torch.rand(pid.shape, device=pid.device) < self.id_drop, torch.zeros_like(pid), pid)
            z = torch.cat([z, self.id_emb(pid)], 1)
        z = z + self.part_emb.weight.repeat(n, 1)
        u = self.phi(z).view(n, 4, -1)
        out = 0
        if self.head in ('ds', 'both'):
            out = out + self.rho(torch.cat([u[:, 0], u[:, 1:].sum(1)], 1))
        if self.head in ('add', 'both'):
            out = out + self.addh(u).sum(1)
        return out


def train_model(model, bt, idx, epochs, lr=2e-3, wd=1e-2, bs=256, seed=0, val=None, eval_every=1, log=print,
                ls=0.0):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * math.ceil(len(idx) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1, anneal_strategy='cos')
    hist = []
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0.0
        perm = rng.permutation(idx)
        for s in range(0, len(perm), bs):
            b = bt.batch(perm[s:s + bs])
            yy = b['y'] * (1 - ls) + 0.5 * ls if ls > 0 else b['y']
            loss = F.binary_cross_entropy_with_logits(model(b), yy)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); tot += loss.item() * b['n']
        msg = f'  ep {ep + 1}/{epochs} loss {tot / len(idx):.4f} ({time.time() - t0:.0f}s)'
        if val is not None and ((ep + 1) % eval_every == 0 or ep == epochs - 1):
            vbt, vai, yv, nnew = val
            p = predict(model, vbt, vai)
            r = eval_val(yv, p, nnew); hist.append((ep + 1, r))
            msg += ' ' + ' '.join(f'{k} {v:.4f}' for k, v in r.items())
        log(msg)
    return model, hist


@torch.no_grad()
def predict(model, bt, ids, bs=1024, logits=True):
    model.eval(); out = []
    for s in range(0, len(ids), bs):
        o = model(bt.batch(ids[s:s + bs]))
        out.append((o if logits else torch.sigmoid(o)).float().cpu().numpy())
    return np.concatenate(out)


def torch_logreg(Xtr, ytr, Xte_list, C=0.03, iters=200, device=None):
    """L2 logistic regression (sklearn objective: C*sum(BCE) + 0.5||w||^2, bias unpenalised), multi-target,
    full-batch L-BFGS in PyTorch on a scipy CSR matrix. Returns decision values for each matrix in Xte_list."""
    dev = device or DEV

    def to_t(X):
        X = X.tocoo()
        return torch.sparse_coo_tensor(np.vstack([X.row, X.col]), X.data.astype(np.float32), X.shape).coalesce().to(dev)
    Xt = to_t(Xtr); Y = torch.from_numpy(np.asarray(ytr, np.float32)).to(dev)
    n, V = Xtr.shape
    W = torch.zeros(V, Y.shape[1], device=dev, requires_grad=True)
    b = torch.full((Y.shape[1],), float(np.log(Y.mean().item() / (1 - Y.mean().item()))), device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn='strong_wolfe',
                            tolerance_grad=1e-6, tolerance_change=1e-9)
    lam = 1.0 / (C * n)

    def closure():
        opt.zero_grad()
        z = torch.sparse.mm(Xt, W) + b
        loss = F.binary_cross_entropy_with_logits(z, Y, reduction='sum') / n + 0.5 * lam * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return [(torch.sparse.mm(to_t(X), W) + b).cpu().numpy() for X in Xte_list]
