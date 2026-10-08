"""Sweep linear models on fingerprint families, 6-fold private-mimic CV. usage: fp_explore.py <spec> [C...]
spec examples: ef3 (ECFP-full r<=3), es4, ef3+ap, pt3, ef3+es3 ; transform: log|bin|raw via env TR."""
import sys, os, pickle, numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib2 import load, make_splits, eval_val
from scipy import sparse
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
_K = None


def keys_of(rec, spec):
    out = []
    for part in spec.split('+'):
        fam = part[:2]
        if fam == 'ap':
            out += rec['ap']
        else:
            r = int(part[2:])
            for rr in range(r + 1):
                out += rec[fam][rr]
    return out


def matrix(recs, spec, vocab=None, min_count=3):
    rows, cols = [], []
    for i, rec in enumerate(recs):
        k = keys_of(rec, spec); rows.append(np.full(len(k), i)); cols.append(np.array(k, np.uint64))
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    if vocab is None:
        u, c = np.unique(cols, return_counts=True); vocab = u[c >= min_count]
    pos = np.clip(np.searchsorted(vocab, cols), 0, len(vocab) - 1); ok = vocab[pos] == cols
    X = sparse.csr_matrix((np.ones(ok.sum(), np.float32), (rows[ok], pos[ok])), shape=(len(recs), len(vocab)))
    X.sum_duplicates()
    return X, vocab


def transform(X, tr):
    X = X.copy()
    if tr == 'log': X.data = np.log1p(X.data)
    elif tr == 'bin': X.data[:] = 1
    elif tr == 'sqrt': X.data = np.sqrt(X.data)
    return X


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


if __name__ == '__main__':
    spec = sys.argv[1]; Cs = [float(c) for c in sys.argv[2:]] or [0.03]
    tr = os.environ.get('TR', 'log'); save = os.environ.get('SAVE', '')
    recs = pickle.load(open(os.path.join(HERE, 'fpkeys.pkl'), 'rb'))['train']
    Fz, y = load(); d = Fz['train']
    Xall, _ = matrix(recs, spec)  # vocab on all train keys (count>=3); per-fold training only uses train rows
    Xall = transform(Xall, tr).tocsr()
    for C in Cs:
        res = []
        for f, (tri, vai, nnew, ns) in enumerate(make_splits(d)):
            P = np.zeros((len(vai), 3))
            for t in range(3):
                m = LogisticRegression(C=C, max_iter=3000).fit(Xall[tri], y[tri, t])
                P[:, t] = logit(m.predict_proba(Xall[vai])[:, 1])
            r = eval_val(y[vai], P, nnew); res.append([r['all'], r['mix'], r['s0'], r['s1'], r['s2'], r['s3']])
            if save:
                os.makedirs(os.path.join(HERE, 'oof'), exist_ok=True)
                np.save(os.path.join(HERE, 'oof', f'{save}_f{f}.npy'), P.astype(np.float32))
        res = np.array(res).mean(0)
        print(f'{spec} tr={tr} C={C} all {res[0]:.4f} mix {res[1]:.4f} s0 {res[2]:.4f} s1 {res[3]:.4f} s2 {res[4]:.4f} s3 {res[5]:.4f}', flush=True)
