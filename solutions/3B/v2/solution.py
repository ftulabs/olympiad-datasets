"""Task 3B v2 - Molecule binding (BRD4 / HSA / sEH): end-to-end solution.

Pipeline (see README.md for the validation + ablations behind every choice)
  1. Data  : parse atoms/typed bonds; hand-made atom features; scaffold + 3 building-block decomposition
             (bridge bonds -> ring systems -> tree centroid); ECFP-like WL keys (r<=3) and atom-pair keys.
  2. Models: (a) GINE GNN with per-part (scaffold / 3 blocks) sum+max pooling and a DeepSets head      ['base']
             (b) same GNN + block-ID embedding with ID-dropout 0.5 and DeepSets+additive heads        ['idboth']
             (c) L2 logistic regression (PyTorch, L-BFGS) on 0/1 presence of ECFP r<=3 keys              ['lr_bin']
             (d) L2 logistic regression on log-counts of ECFP r<=3 + atom-pair (type,type,dist) keys   ['lr_ap']
  3. Train : multi-task BCE, AdamW + one-cycle, epochs fixed on a 6-fold building-block-grouped CV whose
             strata are re-weighted to the private test's novelty mix; refit on all 40k train rows;
             seed ensembles; per-model z-scored logit blend (weights from out-of-fold predictions).
No chemistry libraries, no external data. Device auto-detected (CUDA if available, else CPU).
Usage: python solution.py [--seeds 5] [--workers 4] [--threads 4]
"""
import argparse
import collections
import hashlib
import math
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('DATA3B', '/home/minh/Desktop/olympiad_ai/warmup/3B_molecule_binding/dataset/')
TARGETS = ['bind_BRD4', 'bind_HSA', 'bind_sEH']
DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ============================================================== 1. DATA / FEATURES
ATOMS = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br']
A2I = {a: i for i, a in enumerate(ATOMS)}
BT = {'1': 0, '2': 1, '3': 2, 'a': 3}
BORD = {'1': 1.0, '2': 2.0, '3': 3.0, 'a': 1.5}
VAL = {'C': 4, 'N': 3, 'O': 2, 'S': 2, 'F': 1, 'Cl': 1, 'Br': 1}
RADIUS = 3

def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), 'little')


def bridges(n, adj):
    """Iterative Tarjan: bonds whose removal disconnects the graph (= acyclic bonds)."""
    disc = [-1] * n; low = [0] * n; t = 0; br = set()
    for s in range(n):
        if disc[s] != -1:
            continue
        stack = [(s, -1, iter(adj[s]))]; disc[s] = low[s] = t; t += 1
        while stack:
            u, p, it = stack[-1]
            for v, _ in it:
                if v == p:
                    continue
                if disc[v] == -1:
                    disc[v] = low[v] = t; t += 1; stack.append((v, u, iter(adj[v]))); break
                low[u] = min(low[u], disc[v])
            else:
                stack.pop()
                if p != -1:
                    low[p] = min(low[p], low[u])
                    if low[u] > disc[p]:
                        br.add(frozenset((u, p)))
    return br


def smallest_ring(n, adj, ring_bonds):
    rs = [0] * n
    for u, v in ring_bonds:
        dist = {u: 0}; q = [u]; found = None
        while q and found is None:
            nq = []
            for x in q:
                for y, _ in adj[x]:
                    if (x == u and y == v) or (x == v and y == u):
                        continue
                    if y not in dist:
                        dist[y] = dist[x] + 1
                        if y == v:
                            found = dist[y] + 1; break
                        nq.append(y)
                if found:
                    break
            q = nq
        for a in (u, v):
            if found and (rs[a] == 0 or found < rs[a]):
                rs[a] = found
    return rs


def mol_info(atoms: str, bonds: str) -> dict:
    at = atoms.split(); n = len(at)
    E = [(int(i), int(j), t) for i, j, t in (b.split('-') for b in bonds.split(';'))]
    adj = [[] for _ in range(n)]
    for i, j, t in E:
        adj[i].append((j, t)); adj[j].append((i, t))
    br = bridges(n, adj)
    ring_bonds = [(i, j) for i, j, t in E if frozenset((i, j)) not in br]
    inring = [0] * n
    for i, j in ring_bonds:
        inring[i] = inring[j] = 1
    rs = smallest_ring(n, adj, ring_bonds)
    deg = [len(adj[i]) for i in range(n)]
    nh, arom = [], []
    for i in range(n):
        s = sum(BORD[t] for _, t in adj[i])
        arom.append(int(any(t == 'a' for _, t in adj[i])))
        nh.append(int(max(0, VAL[at[i]] - int(np.floor(s + 1e-6)))) if s <= VAL[at[i]] else 0)
    return dict(at=at, E=E, adj=adj, br=br, inring=inring, rs=rs, deg=deg, nh=nh, arom=arom, n=n)


def atom_feats(m) -> np.ndarray:
    F_ = np.zeros((m['n'], 27), np.uint8)
    for i in range(m['n']):
        F_[i, A2I[m['at'][i]]] = 1
        F_[i, 7 + min(m['deg'][i], 5)] = 1
        F_[i, 13 + min(m['nh'][i], 4)] = 1
        F_[i, 18 + m['arom'][i]] = 1
        F_[i, 20 + m['inring'][i]] = 1
        F_[i, 22 + {0: 0, 3: 1, 4: 1, 5: 2, 6: 3}.get(m['rs'][i], 4)] = 1
    return F_


def ecfp_keys(m, radius=RADIUS) -> np.ndarray:
    """Morgan/ECFP-style WL hashing incl. bond types -> (radius+1, n_atoms) uint64 keys."""
    n = m['n']
    h = [h64(f"{m['at'][i]}|{m['deg'][i]}|{m['nh'][i]}|{m['arom'][i]}|{m['inring'][i]}|{m['rs'][i]}")
         for i in range(n)]
    out = [h]
    for r in range(1, radius + 1):
        h = [h64(f"{r}|{h[i]}|" + ",".join(f"{t}{x}" for t, x in sorted((t, h[j]) for j, t in m['adj'][i])))
             for i in range(n)]
        out.append(h)
    return np.array(out, np.uint64)


def sub_hash(m, mem, mark=()):
    S = set(mem); mark = set(mark)
    h = {i: h64(m['at'][i] + str(m['arom'][i]) + ('*' if i in mark else '')) for i in mem}
    for _ in range(min(len(mem), 8)):
        h = {i: h64(f"{h[i]}|{sorted((t, h[j]) for j, t in m['adj'][i] if j in S)}") for i in mem}
    return h64(str(sorted(h.values())))


def decompose(m):
    """Scaffold + 3 building blocks. Cut bridge bonds -> ring systems; the scaffold is the ring system
    with >=3 external branches that is the tree centroid (3 big branches, smallest max-branch).
    Returns per-atom part id (0 scaffold, 1..3 blocks sorted by block hash), scaffold hash, block hashes."""
    n = m['n']; br = m['br']
    comp = [-1] * n; c = 0
    for s in range(n):
        if comp[s] != -1:
            continue
        comp[s] = c; st = [s]
        while st:
            u = st.pop()
            for v, _ in m['adj'][u]:
                if comp[v] == -1 and frozenset((u, v)) not in br:
                    comp[v] = c; st.append(v)
        c += 1
    groups = collections.defaultdict(list)
    for i, x in enumerate(comp):
        groups[x].append(i)
    best = None
    for mem in groups.values():
        if len(mem) < 3:
            continue
        S = set(mem)
        exts = [(i, j) for i in mem for j, _ in m['adj'][i] if j not in S]
        if len(exts) < 3:
            continue
        subs = []
        for i, j in exts:
            seen = {j}; st = [j]
            while st:
                u = st.pop()
                for v, _ in m['adj'][u]:
                    if v not in S and v not in seen:
                        seen.add(v); st.append(v)
            subs.append((len(seen), i, j, seen))
        subs.sort(key=lambda x: -x[0])
        score = (abs(len([s for s in subs if s[0] >= 3]) - 3), subs[0][0])
        if best is None or score < best[0]:
            best = (score, mem, subs)
    part = np.zeros(n, np.int8)
    if best is None:
        return part, 0, [0, 0, 0]
    _, mem, subs = best
    big = subs[:3]; scaf = set(mem)
    for s in subs[3:]:
        scaf |= s[3]
    sh = sub_hash(m, sorted(scaf), [s[1] for s in big])
    bl = sorted([(sub_hash(m, sorted(s[3]), [s[2]]), s) for s in big], key=lambda x: x[0])
    for k, (_, s) in enumerate(bl):
        for a in s[3]:
            part[a] = k + 1
    bh = [h for h, _ in bl] + [0] * (3 - len(bl))
    return part, sh, bh


def featurize_row(args):
    atoms, bonds = args
    m = mol_info(atoms, bonds)
    part, sh, bh = decompose(m)
    e = np.array([(i, j, BT[t]) for i, j, t in m['E']], np.int16)
    return atom_feats(m), part, e, ecfp_keys(m), sh, bh


def fp_keys(m) -> list:
    """sparse fingerprint keys: ECFP-like (r<=3, full atom invariants) + atom pairs (type_i, type_j, distance)."""
    n = m['n']; adj = m['adj']
    inv = [f"{m['at'][i]}|{m['deg'][i]}|{m['nh'][i]}|{m['arom'][i]}|{m['inring'][i]}|{m['rs'][i]}" for i in range(n)]
    simple = [f"{m['at'][i]}|{m['arom'][i]}|{m['inring'][i]}" for i in range(n)]
    h = [h64('ef' + s) for s in inv]; ef = list(h)
    for r in range(1, 4):
        h = [h64(f"{r}|{h[i]}|" + ",".join(f"{t}{x}" for t, x in sorted((t, h[j]) for j, t in adj[i]))) for i in range(n)]
        ef += h
    dist = np.full((n, n), 99, np.int16)
    for s in range(n):
        dist[s, s] = 0; q = [s]
        while q:
            nq = []
            for u in q:
                for v, _ in adj[u]:
                    if dist[s, v] == 99:
                        dist[s, v] = dist[s, u] + 1; nq.append(v)
            q = nq
    ap = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sorted((simple[i], simple[j]))
            ap.append(h64(f"ap|{a}|{b}|{min(int(dist[i, j]), 15)}"))
    return ef, ap


def featurize_row(args):
    atoms, bonds = args
    m = mol_info(atoms, bonds)
    part, sh, bh = decompose(m)
    e = np.array([(i, j, BT[t]) for i, j, t in m['E']], np.int16)
    ef, ap = fp_keys(m)
    return atom_feats(m), part, e, sh, bh, ef, ap


def featurize(df: pd.DataFrame, workers: int = 2) -> dict:
    rows = list(zip(df.atoms, df.bonds))
    if workers > 1:
        with Pool(workers) as p:
            res = p.map(featurize_row, rows, chunksize=200)
    else:
        res = [featurize_row(r) for r in rows]
    d = dict(na=np.array([len(r[1]) for r in res]), ne=np.array([len(r[2]) for r in res]),
             x=np.concatenate([r[0] for r in res]), part=np.concatenate([r[1] for r in res]),
             e=np.concatenate([r[2] for r in res]),
             scaf=np.array([r[3] for r in res], np.uint64), blocks=np.array([r[4] for r in res], np.uint64),
             ef=[r[5] for r in res], ap=[r[6] for r in res])
    d['ao'] = np.concatenate([[0], np.cumsum(d['na'])]); d['eo'] = np.concatenate([[0], np.cumsum(d['ne'])])
    return d


def fp_matrix(d, use_ap, vocab=None, min_count=3, binary=False):
    """sparse matrix over hashed keys (log1p(count), or 0/1 presence if binary);
    vocabulary = train keys seen >= min_count times."""
    keys = [d['ef'][i] + (d['ap'][i] if use_ap else []) for i in range(len(d['na']))]
    rows = np.concatenate([np.full(len(k), i) for i, k in enumerate(keys)])
    cols = np.concatenate([np.array(k, np.uint64) for k in keys])
    if vocab is None:
        u, c = np.unique(cols, return_counts=True); vocab = u[c >= min_count]
    pos = np.clip(np.searchsorted(vocab, cols), 0, len(vocab) - 1); ok = vocab[pos] == cols
    X = sparse.csr_matrix((np.ones(ok.sum(), np.float32), (rows[ok], pos[ok])), shape=(len(keys), len(vocab)))
    X.sum_duplicates(); X.data = np.ones_like(X.data) if binary else np.log1p(X.data)
    return X, vocab


# ============================================================== 2. MODELS
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


# ============================================================== 3. TRAINING & INFERENCE


def train_model(model, bt, idx, epochs, lr=2e-3, wd=1e-2, bs=256, seed=0, log=print, ls=0.0):
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


def zs(p):
    return (p - p.mean(0)) / (p.std(0) + 1e-9)


GNN_CFGS = [  # (family, Net kwargs, epochs, seeds); members of a family are averaged before blending
    ('base', dict(head='ds', ids=False), 25, 5),
    ('base', dict(head='ds', ids=False), 40, 3),
    ('idboth', dict(head='both', ids=True), 25, 5),
    ('idboth', dict(head='both', ids=True), 40, 2),
]
BLEND_W = {'base': 2.0, 'idboth': 1.5, 'lr_bin': 1.0, 'lr_ap': 0.5}  # chosen on 6-fold OOF (README)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', default=HERE)
    ap.add_argument('--save_parts', default='')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    print('device', DEV, flush=True)
    t0 = time.time()
    tr = pd.read_csv(DATA + 'train/train.csv')
    tests = {'public': pd.read_csv(DATA + 'public_test/public_test.csv'),
             'private': pd.read_csv(DATA + 'private_test/private_test.csv')}
    y = tr[TARGETS].values.astype(np.float32)
    dtr = featurize(tr, a.workers)
    dte = {k: featurize(v, a.workers) for k, v in tests.items()}
    print(f'features done in {time.time() - t0:.0f}s', flush=True)

    preds = {k: {} for k in tests}
    # --- linear fingerprint models (PyTorch L-BFGS logistic regression)
    for name, use_ap, binary, C in [('lr_bin', False, True, 0.03), ('lr_ap', True, False, 0.03)]:
        X, voc = fp_matrix(dtr, use_ap, binary=binary)
        outs = torch_logreg(X, y, [fp_matrix(dte[k], use_ap, voc, binary=binary)[0] for k in tests], C=C)
        for k, o in zip(tests, outs):
            preds[k][name] = o
        print(f'{name} done ({time.time() - t0:.0f}s)', flush=True)
    # --- GNNs, seed ensembles, fit on all labelled rows
    idv = np.unique(np.concatenate([dtr['scaf'], dtr['blocks'].ravel()]))
    bt = Batcher(dtr, idv, y)
    bte = {k: Batcher(v, idv) for k, v in dte.items()}
    acc = {k: collections.defaultdict(list) for k in tests}
    for fam, kw, ep, n_seeds in GNN_CFGS:
        for s in range(min(n_seeds, a.seeds)):
            print(f'GNN {fam} {ep} epochs, seed {s}', flush=True)
            model = Net(n_ids=len(idv) if kw['ids'] else 0, head=kw['head'])
            model, _ = train_model(model, bt, np.arange(len(y)), ep, seed=s, log=lambda m: print(m, flush=True))
            for k in tests:
                acc[k][fam].append(predict(model, bte[k], np.arange(len(tests[k]))))
    for k in tests:
        for fam, lst in acc[k].items():
            preds[k][fam] = np.mean(lst, 0)
    # --- blend: z-scored logits, fixed weights
    for k, df in tests.items():
        if a.save_parts:
            np.savez(os.path.join(a.save_parts, f'parts_{k}.npz'), **preds[k])
        z = sum(BLEND_W[m] * zs(p) for m, p in preds[k].items()) / sum(BLEND_W[m] for m in preds[k])
        sub = pd.DataFrame(1 / (1 + np.exp(-z)), columns=TARGETS)
        sub.insert(0, 'id', df['id'].values)
        sub.to_csv(os.path.join(a.out, f'{k}_submission.csv'), index=False)
    print(f'done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
