import os, sys, pickle, collections, time, math
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import average_precision_score
torch.set_num_threads(int(os.environ.get('NTHREADS','2')))
DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
W = os.path.dirname(os.path.abspath(__file__))
D = os.environ.get('DATA3B', '/home/minh/Desktop/olympiad_ai/warmup/3B_molecule_binding/dataset/')
T = ['bind_BRD4', 'bind_HSA', 'bind_sEH']


def mean_ap(y, p):
    return float(np.mean([average_precision_score(y[:, k], p[:, k]) for k in range(3)]))


def load():
    z = np.load(f'{W}/cache.npz')
    res = {}
    for k in ['tr', 'pu', 'pr']:
        d = {f: z[f'{k}_{f}'] for f in ['na', 'ne', 'x', 'part', 'e', 'keys', 'scaf', 'blocks']}
        d['ao'] = np.concatenate([[0], np.cumsum(d['na'])]); d['eo'] = np.concatenate([[0], np.cumsum(d['ne'])])
        res[k] = d
    y = pd.read_csv(D + 'train/train.csv')[T].values.astype(np.float32)
    return res, y


def build_vocab(d, mol_ids, min_count=3, max_radius=3):
    """keys (all radii <= max_radius) present in >= min_count of the given molecules"""
    ks = []
    for i in mol_ids:
        ks.append(np.unique(d['keys'][:max_radius + 1, d['ao'][i]:d['ao'][i + 1]]))
    u, c = np.unique(np.concatenate(ks), return_counts=True)
    return u[c >= min_count]


def key_ids(d, vocab, max_radius=3):
    K = d['keys'][:max_radius + 1].T  # (atoms, R+1)
    pos = np.searchsorted(vocab, K); pos = np.clip(pos, 0, len(vocab) - 1)
    ok = vocab[pos] == K
    return np.where(ok, pos + 1, 0).astype(np.int64)


def make_splits(res, y, n_folds=3, block_frac=1 / 8, seed=0):
    """grouped, novelty-aware splits: hold out a random 1/8 of building blocks + one scaffold."""
    rng = np.random.RandomState(seed)
    d = res['tr']
    blocks = np.unique(d['blocks']); scafs = np.unique(d['scaf'])
    rng.shuffle(blocks); rng.shuffle(scafs)
    nb = int(len(blocks) * block_frac)
    splits = []
    for f in range(n_folds):
        hb = blocks[f * nb:(f + 1) * nb]
        nnew = np.isin(d['blocks'], hb).sum(1) + (d['scaf'] == scafs[f % len(scafs)])
        tri = np.where(nnew == 0)[0]; vai = np.where(nnew > 0)[0]
        splits.append((tri, vai, nnew[vai]))
    return splits


def eval_split(y, p, nnew):
    r = {'all': mean_ap(y, p)}
    m2 = nnew >= 2
    r['ge2'] = mean_ap(y[m2], p[m2])
    return r


def id_vocab(d, mol_ids):
    return np.unique(np.concatenate([d['scaf'][mol_ids], d['blocks'][mol_ids].ravel()]))


class Batcher:
    def __init__(self, d, vocab, y=None, max_radius=3, idv=None):
        self.d = d; self.y = y
        H = np.concatenate([d['scaf'][:, None], d['blocks']], 1)  # (mols, 4)
        if idv is None: idv = np.zeros(1, np.uint64)
        pos = np.clip(np.searchsorted(idv, H), 0, len(idv) - 1)
        self.pid = np.where(idv[pos] == H, pos + 1, 0).astype(np.int64)
        self.kid = key_ids(d, vocab, max_radius)

    def batch(self, ids):
        d = self.d
        aidx = np.concatenate([np.arange(d['ao'][i], d['ao'][i + 1]) for i in ids])
        eidx = np.concatenate([np.arange(d['eo'][i], d['eo'][i + 1]) for i in ids])
        na = d['na'][ids]; ne = d['ne'][ids]
        loc_off = np.repeat(np.concatenate([[0], np.cumsum(na)[:-1]]), ne)
        e = d['e'][eidx].astype(np.int64)
        s = e[:, 0] + loc_off; t = e[:, 1] + loc_off
        bi = np.repeat(np.arange(len(ids)), na)
        xb = np.concatenate([d['x'][aidx].astype(np.float32), np.eye(4, dtype=np.float32)[d['part'][aidx]]], 1)
        b = dict(x=torch.from_numpy(xb),
                 src=torch.from_numpy(np.concatenate([s, t])), dst=torch.from_numpy(np.concatenate([t, s])),
                 et=torch.from_numpy(np.concatenate([e[:, 2], e[:, 2]])), bi=torch.from_numpy(bi),
                 pi=torch.from_numpy(bi * 4 + d['part'][aidx].astype(np.int64)), n=len(ids),
                 kid=torch.from_numpy(self.kid[aidx]), pid=torch.from_numpy(self.pid[ids].ravel()))
        if self.y is not None:
            b['y'] = torch.from_numpy(self.y[ids])
        return {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}


# ----------------------------- models -----------------------------
def scatter_sum(h, idx, n):
    return torch.zeros(n, h.size(1), dtype=h.dtype, device=h.device).index_add_(0, idx, h)


def scatter_max(h, idx, n):
    out = torch.full((n, h.size(1)), -1e4, dtype=h.dtype, device=h.device)
    return out.scatter_reduce(0, idx.unsqueeze(1).expand_as(h), h, 'amax', include_self=True)


class GINE(nn.Module):
    def __init__(self, din, hid=128, layers=4, drop=0.1):
        super().__init__()
        self.inp = nn.Linear(din, hid)
        self.eemb = nn.ModuleList([nn.Embedding(4, hid) for _ in range(layers)])
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(hid, 2 * hid), nn.BatchNorm1d(2 * hid), nn.ReLU(),
                                                 nn.Linear(2 * hid, hid)) for _ in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hid) for _ in range(layers)])
        self.eps = nn.Parameter(torch.zeros(layers))
        self.drop = drop

    def forward(self, b):
        h = self.inp(b['x'])
        for l in range(len(self.mlps)):
            msg = F.relu(h[b['src']] + self.eemb[l](b['et']))
            agg = scatter_sum(msg, b['dst'], h.size(0))
            z = self.mlps[l]((1 + self.eps[l]) * h + agg)
            z = F.dropout(F.relu(self.bns[l](z)), self.drop, self.training)
            h = h + z
        return h


class Net(nn.Module):
    """mode: 'fp' (whole-molecule fingerprint MLP), 'ds' (DeepSets over scaffold+3 blocks fingerprints),
    'gnn' (GINE, mol pooling), 'gnnds' (GINE with per-part pooling + DeepSets), 'hyb' (gnnds + ds fingerprint)."""

    def __init__(self, mode, vocab_size, din=31, hid=128, layers=4, emb=128, drop=0.2, n_ids=0, id_drop=0.5):
        super().__init__()
        self.n_ids = n_ids; self.id_drop = id_drop
        self.mode = mode
        self.use_fp = mode in ('fp', 'ds', 'hyb')
        self.use_gnn = mode in ('gnn', 'gnnds', 'hyb')
        self.per_part = mode in ('ds', 'gnnds', 'hyb')
        d = 0
        if self.use_fp:
            self.bag = nn.Embedding(vocab_size + 1, emb, padding_idx=0)
            nn.init.normal_(self.bag.weight, std=0.1)
            with torch.no_grad(): self.bag.weight[0].zero_()
            d += emb
        if self.use_gnn:
            self.gnn = GINE(din, hid, layers)
            d += 2 * hid
        if n_ids > 0:
            self.id_emb = nn.Embedding(n_ids + 1, emb, padding_idx=0)
            nn.init.normal_(self.id_emb.weight, std=0.1)
            with torch.no_grad(): self.id_emb.weight[0].zero_()
            d += emb
        self.part_emb = nn.Embedding(4, d)
        self.phi = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(2 * hid, hid), nn.ReLU())
        self.additive = False
        self.head_add = nn.Linear(hid, 3)
        self.rho = nn.Sequential(nn.Dropout(drop), nn.Linear(hid * (2 if self.per_part else 1), hid), nn.ReLU(), nn.Linear(hid, 3))

    def forward(self, b):
        n = b['n']; feats = []
        if self.use_fp:
            pe = scatter_sum(self.bag(b['kid']).sum(1), b['pi'], 4 * n)  # (4n, emb) per-part fingerprint sum
            feats.append(pe)
        if self.use_gnn:
            h = self.gnn(b)
            feats.append(torch.cat([scatter_sum(h, b['pi'], 4 * n), scatter_max(h, b['pi'], 4 * n).clamp(min=-50)], 1))
        if self.n_ids > 0:
            pid = b['pid']
            if self.training and self.id_drop > 0:
                pid = torch.where(torch.rand(pid.shape, device=pid.device) < self.id_drop, torch.zeros_like(pid), pid)
            feats.append(self.id_emb(pid))
        z = torch.cat(feats, 1)  # (4n, d) per part
        if self.per_part:
            z = z + self.part_emb.weight.repeat(n, 1)
            u = self.phi(z).view(n, 4, -1)
            if self.additive:
                return self.head_add(u).sum(1)
            pooled = torch.cat([u[:, 0], u[:, 1:].sum(1)], 1)  # scaffold, sum over blocks (perm-invariant)
        else:
            zz = z.view(n, 4, -1).sum(1)
            pooled = self.phi(zz)
        return self.rho(pooled)


def predict(model, bt, idx, bs=1024):
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            out.append(torch.sigmoid(model(bt.batch(idx[s:s + bs]))).cpu().numpy())
    return np.concatenate(out)


def train(model, bt, tri, epochs=20, bs=256, lr=2e-3, wd=1e-4, pos_weight=None, val=None, log=print, seed=0,
          focal=0.0):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * math.ceil(len(tri) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1, anneal_strategy='cos')
    pw = torch.tensor(pos_weight, dtype=torch.float32, device=DEV) if pos_weight is not None else None
    hist = []; best = (-1, None, -1)
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0
        perm = rng.permutation(tri)
        for s in range(0, len(perm), bs):
            b = bt.batch(perm[s:s + bs])
            logit = model(b)
            loss = F.binary_cross_entropy_with_logits(logit, b['y'], pos_weight=pw, reduction='none')
            if focal > 0:
                p = torch.sigmoid(logit); pt = torch.where(b['y'] > 0, p, 1 - p)
                loss = loss * (1 - pt).pow(focal)
            loss = loss.mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
            tot += loss.item() * len(b['y'])
        msg = f'ep {ep + 1} loss {tot / len(tri):.4f} {time.time() - t0:.0f}s'
        if val is not None:
            vai, yv, nnew = val
            p = predict(model, bt, vai)
            r = eval_split(yv, p, nnew); hist.append(r)
            msg += f" val {r['all']:.4f} ge2 {r['ge2']:.4f}"
            if r['all'] > best[0]:
                best = (r['all'], p, ep + 1)
        log(msg)
    return hist, best
