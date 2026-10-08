"""Extra sparse fingerprint families (no chemistry libs): ECFP r0-5 with two atom-invariant sets,
atom-pair (type_i, type_j, topological distance) and part-tagged keys. Output: dict of per-molecule key lists."""
import sys, os, pickle, collections
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1sol as S

D = os.environ.get('DATA3B', '/home/minh/Desktop/olympiad_ai/warmup/3B_molecule_binding/dataset/')


def row_keys(args):
    atoms, bonds = args
    m = S.mol_info(atoms, bonds)
    part, _, _ = S.decompose(m)
    n = m['n']; adj = m['adj']
    out = {}
    inv_full = [f"{m['at'][i]}|{m['deg'][i]}|{m['nh'][i]}|{m['arom'][i]}|{m['inring'][i]}|{m['rs'][i]}" for i in range(n)]
    inv_simple = [f"{m['at'][i]}|{m['arom'][i]}|{m['inring'][i]}" for i in range(n)]
    for name, inv in [('ef', inv_full), ('es', inv_simple)]:
        h = [S.h64(name + s) for s in inv]; keys = [list(h)]
        for r in range(1, 6):
            h = [S.h64(f"{r}|{h[i]}|" + ",".join(f"{t}{x}" for t, x in sorted((t, h[j]) for j, t in adj[i]))) for i in range(n)]
            keys.append(list(h))
        out[name] = keys  # list over radius of per-atom keys
    # part-tagged ECFP-full (scaffold vs block) radius<=3
    out['pt'] = [[S.h64(f"{int(part[i] > 0)}|{k}") for i, k in enumerate(out['ef'][r])] for r in range(4)]
    # atom pairs with topological distance (BFS), atom type = element|arom|inring
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
            a, b = sorted((inv_simple[i], inv_simple[j]))
            ap.append(S.h64(f"ap|{a}|{b}|{min(int(dist[i, j]), 15)}"))
    out['ap'] = ap
    return out


def build(workers=4):
    res = {}
    for k, f in [('train', 'train/train.csv'), ('public', 'public_test/public_test.csv'), ('private', 'private_test/private_test.csv')]:
        df = pd.read_csv(D + f)
        with Pool(workers) as p:
            res[k] = p.map(row_keys, list(zip(df.atoms, df.bonds)), chunksize=200)
    return res


if __name__ == '__main__':
    r = build(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
    pickle.dump(r, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fpkeys.pkl'), 'wb'))
