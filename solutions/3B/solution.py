"""Task 3B - Molecule binding (BRD4 / HSA / sEH), end-to-end solution.

Pipeline
  1. Data   : parse atoms + typed bonds, hand-made atom features (element, degree, implicit H, aromatic,
              ring membership, smallest ring size), ECFP-like WL hashes with bond types, and a
              scaffold + 3-building-block decomposition (bridge bonds -> ring systems -> tree centroid).
  2. Model  : (a) GINE GNN (edge-type-aware messages, residual + BatchNorm) whose atom states are pooled
              PER PART (scaffold, block1..3; sum+max) and combined DeepSets-style (scaffold, sum of blocks);
              (b) fingerprint DeepSets model (per-part ECFP r<=2 embedding bag + block-ID embedding with
              ID-dropout so the model learns to fall back on structure for unseen blocks).
  3. Train  : multi-task BCE, AdamW + one-cycle cosine, fixed epoch budget chosen on a grouped
              (held-out building blocks + held-out scaffold) validation, seed ensemble, fit on all train.

Usage:  python solution.py [--gnn_seeds 3] [--fp_seeds 3] [--workers 2]
Writes public_submission.csv / private_submission.csv next to this file.
No chemistry libraries, no external data. Runs on CPU or CUDA (auto-detected).
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


def featurize(df: pd.DataFrame, workers: int = 2) -> dict:
    rows = list(zip(df.atoms, df.bonds))
    if workers > 1:
        with Pool(workers) as p:
            res = p.map(featurize_row, rows, chunksize=500)
    else:
        res = [featurize_row(r) for r in rows]
    d = dict(na=np.array([len(r[1]) for r in res]), ne=np.array([len(r[2]) for r in res]),
             x=np.concatenate([r[0] for r in res]), part=np.concatenate([r[1] for r in res]),
             e=np.concatenate([r[2] for r in res]), keys=np.concatenate([r[3] for r in res], 1),
             scaf=np.array([r[4] for r in res], np.uint64), blocks=np.array([r[5] for r in res], np.uint64))
    d['ao'] = np.concatenate([[0], np.cumsum(d['na'])]); d['eo'] = np.concatenate([[0], np.cumsum(d['ne'])])
    return d


def build_vocab(d, min_count=3, max_radius=2):
    ks = [np.unique(d['keys'][:max_radius + 1, d['ao'][i]:d['ao'][i + 1]]) for i in range(len(d['na']))]
    u, c = np.unique(np.concatenate(ks), return_counts=True)
    return u[c >= min_count]


def lookup(vocab, K):
    pos = np.clip(np.searchsorted(vocab, K), 0, len(vocab) - 1)
    return np.where(vocab[pos] == K, pos + 1, 0).astype(np.int64)


class Batcher:
    """Packs many molecules into one disconnected graph; also per-part fingerprint key ids and part ids."""

    def __init__(self, d, vocab, idv, y=None, max_radius=2):
        self.d = d; self.y = y
        self.kid = lookup(vocab, d['keys'][:max_radius + 1].T)
        self.pid = lookup(idv, np.concatenate([d['scaf'][:, None], d['blocks']], 1))

    def batch(self, ids):
        d = self.d
        aidx = np.concatenate([np.arange(d['ao'][i], d['ao'][i + 1]) for i in ids])
        eidx = np.concatenate([np.arange(d['eo'][i], d['eo'][i + 1]) for i in ids])
        na = d['na'][ids]; ne = d['ne'][ids]
        off = np.repeat(np.concatenate([[0], np.cumsum(na)[:-1]]), ne)
        e = d['e'][eidx].astype(np.int64)
        s = e[:, 0] + off; t = e[:, 1] + off
        bi = np.repeat(np.arange(len(ids)), na)
        x = np.concatenate([d['x'][aidx].astype(np.float32), np.eye(4, dtype=np.float32)[d['part'][aidx]]], 1)
        b = dict(x=x, src=np.concatenate([s, t]), dst=np.concatenate([t, s]), et=np.concatenate([e[:, 2], e[:, 2]]),
                 pi=bi * 4 + d['part'][aidx].astype(np.int64), kid=self.kid[aidx], pid=self.pid[ids].ravel())
        if self.y is not None:
            b['y'] = self.y[ids]
        b = {k: torch.from_numpy(v).to(DEV) for k, v in b.items()}
        b['n'] = len(ids)
        return b


# ============================================================== 2. MODEL
def scatter_sum(h, idx, n):
    return torch.zeros(n, h.size(1), dtype=h.dtype, device=h.device).index_add_(0, idx, h)


def scatter_max(h, idx, n):
    out = torch.full((n, h.size(1)), -1e4, dtype=h.dtype, device=h.device)
    return out.scatter_reduce(0, idx.unsqueeze(1).expand_as(h), h, 'amax', include_self=True)


class GINE(nn.Module):
    """GIN with edge-type embeddings added to messages (GINE), residual connections + BatchNorm."""

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
            z = self.mlps[l]((1 + self.eps[l]) * h + scatter_sum(msg, b['dst'], h.size(0)))
            h = h + F.dropout(F.relu(self.bns[l](z)), self.drop, self.training)
        return h


class Net(nn.Module):
    """mode 'gnnds': GINE -> per-part sum/max pooling -> DeepSets(scaffold, sum of blocks) -> 3 logits.
    mode 'ds'   : per-part ECFP embedding-bag (+ block-ID embedding w/ ID dropout) -> DeepSets -> 3 logits."""

    def __init__(self, mode, vocab_size, n_ids=0, din=31, hid=128, layers=4, emb=128, drop=0.3, id_drop=0.5):
        super().__init__()
        self.mode, self.n_ids, self.id_drop = mode, n_ids, id_drop
        d = 0
        if mode == 'ds':
            self.bag = nn.Embedding(vocab_size + 1, emb, padding_idx=0)
            nn.init.normal_(self.bag.weight, std=0.1)
            d += emb
        else:
            self.gnn = GINE(din, hid, layers)
            d += 2 * hid
        if n_ids > 0:
            self.id_emb = nn.Embedding(n_ids + 1, emb, padding_idx=0)
            nn.init.normal_(self.id_emb.weight, std=0.1)
            d += emb
        with torch.no_grad():
            for e in [getattr(self, 'bag', None), getattr(self, 'id_emb', None)]:
                if e is not None:
                    e.weight[0].zero_()
        self.part_emb = nn.Embedding(4, d)
        self.phi = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * hid), nn.ReLU(), nn.Dropout(drop),
                                 nn.Linear(2 * hid, hid), nn.ReLU())
        self.rho = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * hid, hid), nn.ReLU(), nn.Linear(hid, 3))

    def forward(self, b):
        n = b['n']; feats = []
        if self.mode == 'ds':
            feats.append(scatter_sum(self.bag(b['kid']).sum(1), b['pi'], 4 * n))
        else:
            h = self.gnn(b)
            feats.append(torch.cat([scatter_sum(h, b['pi'], 4 * n), scatter_max(h, b['pi'], 4 * n).clamp(min=-50)], 1))
        if self.n_ids > 0:
            pid = b['pid']
            if self.training and self.id_drop > 0:
                pid = torch.where(torch.rand(pid.shape, device=pid.device) < self.id_drop, torch.zeros_like(pid), pid)
            feats.append(self.id_emb(pid))
        z = torch.cat(feats, 1) + self.part_emb.weight.repeat(n, 1)
        u = self.phi(z).view(n, 4, -1)
        return self.rho(torch.cat([u[:, 0], u[:, 1:].sum(1)], 1))  # permutation-invariant over blocks


# ============================================================== 3. TRAINING & INFERENCE
def train_model(model, bt, idx, epochs, lr=2e-3, wd=1e-2, bs=256, seed=0, log=print):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * math.ceil(len(idx) / bs),
                                                pct_start=0.1, anneal_strategy='cos')
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0.0
        perm = rng.permutation(idx)
        for s in range(0, len(perm), bs):
            b = bt.batch(perm[s:s + bs])
            loss = F.binary_cross_entropy_with_logits(model(b), b['y'])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); tot += loss.item() * b['n']
        log(f'  ep {ep + 1}/{epochs} loss {tot / len(idx):.4f} ({time.time() - t0:.0f}s)')
    return model


@torch.no_grad()
def predict(model, bt, n, bs=1024):
    model.eval()
    return np.concatenate([torch.sigmoid(model(bt.batch(np.arange(s, min(s + bs, n))))).cpu().numpy()
                           for s in range(0, n, bs)])


def rank01(p):
    """per-column rank transform to [0,1] (AP only depends on ranking; makes blending scale-free)."""
    return np.argsort(np.argsort(p, 0), 0) / (len(p) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gnn_seeds', type=int, default=6)
    ap.add_argument('--fp_seeds', type=int, default=3)
    ap.add_argument('--gnn_epochs', type=int, default=15)
    ap.add_argument('--fp_epochs', type=int, default=7)
    ap.add_argument('--w_gnn', type=float, default=0.8)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--out', default=HERE)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    print('device', DEV)

    t0 = time.time()
    tr = pd.read_csv(DATA + 'train/train.csv')
    tests = {'public': pd.read_csv(DATA + 'public_test/public_test.csv'),
             'private': pd.read_csv(DATA + 'private_test/private_test.csv')}
    y = tr[TARGETS].values.astype(np.float32)
    dtr = featurize(tr, a.workers)
    dte = {k: featurize(v, a.workers) for k, v in tests.items()}
    print(f'features done in {time.time() - t0:.0f}s')

    vocab = build_vocab(dtr, 3, 2)
    idv = np.unique(np.concatenate([dtr['scaf'], dtr['blocks'].ravel()]))  # known scaffold/block ids
    bt = Batcher(dtr, vocab, idv, y)
    bte = {k: Batcher(v, vocab, idv) for k, v in dte.items()}
    preds = {k: {'gnn': [], 'fp': []} for k in tests}
    all_idx = np.arange(len(y))
    for s in range(a.gnn_seeds):
        print(f'GNN-DeepSets seed {s}')
        m = train_model(Net('gnnds', len(vocab)), bt, all_idx, a.gnn_epochs, seed=100 + s)
        for k in tests:
            preds[k]['gnn'].append(predict(m, bte[k], len(tests[k])))
    for s in range(a.fp_seeds):
        print(f'FP-DeepSets(+ID) seed {s}')
        m = train_model(Net('ds', len(vocab), n_ids=len(idv)), bt, all_idx, a.fp_epochs, seed=200 + s)
        for k in tests:
            preds[k]['fp'].append(predict(m, bte[k], len(tests[k])))

    for k, df in tests.items():
        parts = []
        if preds[k]['gnn']:
            parts.append((a.w_gnn, rank01(np.mean(preds[k]['gnn'], 0))))
        if preds[k]['fp']:
            parts.append((1 - a.w_gnn, rank01(np.mean(preds[k]['fp'], 0))))
        p = sum(w * q for w, q in parts) / sum(w for w, _ in parts)
        sub = pd.DataFrame(np.clip(p, 0, 1), columns=TARGETS)
        sub.insert(0, 'id', df['id'].values)
        sub.to_csv(os.path.join(a.out, f'{k}_submission.csv'), index=False)
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
