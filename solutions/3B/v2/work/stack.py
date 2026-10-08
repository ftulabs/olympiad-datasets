"""Blend + per-novelty-stratum calibration on 6-fold OOF logits. usage: stack.py tagA[:w] tagB[:w] ...
Prints per-model and blended metrics (mean over folds), and leave-one-fold-out calibrated metrics."""
import sys, os, numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib2 import load, make_splits, eval_val
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
Fz, y = load(); d = Fz['train']
SPL = make_splits(d)


def get(tag, f):
    p = os.path.join(HERE, 'oof', f'{tag}_f{f}.npy')
    return np.load(p) if os.path.exists(p) else None


def zs(p):
    return (p - p.mean(0)) / (p.std(0) + 1e-9)


def blend(tags_w, f):
    ps = [(w, get(t, f)) for t, w in tags_w]
    if any(p is None for _, p in ps): return None
    return sum(w * zs(p) for w, p in ps) / sum(w for w, _ in ps)


def strat_feats(p, nnew, ns):
    S = np.zeros((len(nnew), 5)); S[np.arange(len(nnew)), nnew] = 1; S[:, 4] = ns
    return S


def stack_eval(tags_w, folds, C=1.0):
    """leave-one-fold-out stacker: per target LR on [z_m * S for each model m] + S (stratum-specific weights)."""
    data = {}
    for f in folds:
        tri, vai, nnew, ns = SPL[f]
        data[f] = ([zs(get(t, f)) for t, _ in tags_w], y[vai], nnew, ns.astype(float))
    def feats(ps, nnew, ns, t):
        S = strat_feats(None, nnew, ns)
        return np.concatenate([p[:, [t]] * S for p in ps] + [p[:, [t]] for p in ps] + [S], 1)
    out = []
    for f in folds:
        ps, yv, nnew, ns = data[f]; P = np.zeros((len(yv), 3))
        for t in range(3):
            X = np.concatenate([feats(data[g][0], data[g][2], data[g][3], t) for g in folds if g != f])
            Y = np.concatenate([data[g][1][:, t] for g in folds if g != f])
            m = LogisticRegression(C=C, max_iter=3000).fit(X, Y)
            P[:, t] = m.decision_function(feats(ps, nnew, ns, t))
        out.append(eval_val(yv, P, nnew))
    return out


def calibrate_eval(tags_w, folds):
    """leave-one-fold-out: fit per-target LR on [z, z*S, S] from other folds' OOF, apply to held fold."""
    data = {}
    for f in folds:
        tri, vai, nnew, ns = SPL[f]; p = blend(tags_w, f)
        data[f] = (p, y[vai], nnew, ns.astype(float))
    res_raw, res_cal = [], []
    for f in folds:
        p, yv, nnew, ns = data[f]
        P = np.zeros_like(p)
        for t in range(3):
            Xs, Ys = [], []
            for g in folds:
                if g == f: continue
                pg, yg, ng, sg = data[g]; S = strat_feats(pg, ng, sg)
                Xs.append(np.concatenate([pg[:, [t]], pg[:, [t]] * S, S], 1)); Ys.append(yg[:, t])
            m = LogisticRegression(C=1.0, max_iter=2000).fit(np.concatenate(Xs), np.concatenate(Ys))
            S = strat_feats(p, nnew, ns)
            P[:, t] = m.decision_function(np.concatenate([p[:, [t]], p[:, [t]] * S, S], 1))
        res_raw.append(eval_val(yv, p, nnew)); res_cal.append(eval_val(yv, P, nnew))
    return res_raw, res_cal


def summarize(rs):
    keys = ['all', 'mix', 's0', 's1', 's2', 's3']
    return ' '.join(f'{k} {np.mean([r.get(k, np.nan) for r in rs]):.4f}' for k in keys)


if __name__ == '__main__':
    specs = sys.argv[1:]
    tags_w = [(s.split(':')[0], float(s.split(':')[1]) if ':' in s else 1.0) for s in specs]
    folds = [f for f in range(len(SPL)) if all(get(t, f) is not None for t, _ in tags_w)]
    print('folds', folds)
    for t, _ in tags_w:
        rs = [eval_val(y[SPL[f][1]], get(t, f), SPL[f][2]) for f in folds]
        print(f'{t:>14}: {summarize(rs)}')
    if len(tags_w) > 1:
        rs = [eval_val(y[SPL[f][1]], blend(tags_w, f), SPL[f][2]) for f in folds]
        print(f'{"BLEND":>14}: {summarize(rs)}')
    raw, cal = calibrate_eval(tags_w, folds)
    print(f'{"CALIBRATED":>14}: {summarize(cal)}')
    if len(tags_w) > 1:
        print(f'{"STACKED":>14}: {summarize(stack_eval(tags_w, folds))}')
