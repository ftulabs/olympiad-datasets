import sys, argparse, json
sys.path.insert(0, __import__('os').path.dirname(__file__))
from lib import *
ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='fp'); ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--epochs', type=int, default=15); ap.add_argument('--hid', type=int, default=128)
ap.add_argument('--layers', type=int, default=4); ap.add_argument('--lr', type=float, default=2e-3)
ap.add_argument('--wd', type=float, default=1e-4); ap.add_argument('--drop', type=float, default=0.2)
ap.add_argument('--radius', type=int, default=3); ap.add_argument('--minc', type=int, default=3)
ap.add_argument('--pw', type=float, default=0); ap.add_argument('--focal', type=float, default=0)
ap.add_argument('--tag', default=''); ap.add_argument('--ids', type=int, default=0); ap.add_argument('--id_drop', type=float, default=0.5)
ap.add_argument('--emb', type=int, default=128); ap.add_argument('--add', type=int, default=0); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
res, y = load()
spl = make_splits(res, y)[a.fold]
tri, vai, nnew = spl
vocab = build_vocab(res['tr'], tri, a.minc, a.radius)
idv = id_vocab(res['tr'], tri)
bt = Batcher(res['tr'], vocab, y, a.radius, idv)
del res
print(f'train {len(tri)} val {len(vai)} vocab {len(vocab)} nnew dist {np.bincount(nnew)} valpos {y[vai].mean(0)}', flush=True)
model = Net(a.mode, len(vocab), hid=a.hid, layers=a.layers, drop=a.drop, emb=a.emb, n_ids=len(idv) if a.ids else 0, id_drop=a.id_drop)
model.additive = bool(a.add); model.to(DEV)
pw = [a.pw] * 3 if a.pw > 0 else None
hist, best = train(model, bt, tri, a.epochs, lr=a.lr, wd=a.wd, pos_weight=pw, val=(vai, y[vai], nnew),
                   log=lambda s: print(s, flush=True), focal=a.focal, seed=a.seed)
print('BEST', json.dumps(dict(vars(a), best=best[0], best_ep=best[2], last=hist[-1])), flush=True)
np.save(f'{W}/oof_{a.mode}{a.tag}_f{a.fold}.npy', best[1])
