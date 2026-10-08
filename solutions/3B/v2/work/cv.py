import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib2 import *

ap = argparse.ArgumentParser()
ap.add_argument('--fold', type=int, default=0)  # -1 = fit on all train, predict tests
ap.add_argument('--epochs', type=int, default=15); ap.add_argument('--hid', type=int, default=128)
ap.add_argument('--layers', type=int, default=4); ap.add_argument('--lr', type=float, default=2e-3)
ap.add_argument('--wd', type=float, default=1e-2); ap.add_argument('--drop', type=float, default=0.3)
ap.add_argument('--gdrop', type=float, default=0.1); ap.add_argument('--ids', type=int, default=0)
ap.add_argument('--id_drop', type=float, default=0.5); ap.add_argument('--emb', type=int, default=64)
ap.add_argument('--head', default='ds'); ap.add_argument('--cut', type=int, default=0)
ap.add_argument('--jk', type=int, default=0); ap.add_argument('--bs', type=int, default=256)
ap.add_argument('--ls', type=float, default=0.0)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--tag', default='x')
ap.add_argument('--eval_every', type=int, default=3); ap.add_argument('--threads', type=int, default=2)
a = ap.parse_args()
torch.set_num_threads(a.threads)
Fz, y = load(); d = Fz['train']
os.makedirs(os.path.join(HERE, 'oof'), exist_ok=True)
kw = dict(din=32, hid=a.hid, layers=a.layers, drop=a.drop, gdrop=a.gdrop, emb=a.emb, id_drop=a.id_drop,
          head=a.head, jk=bool(a.jk))
if a.fold >= 0:
    tri, vai, nnew, ns = make_splits(d)[a.fold]
    idv = np.unique(np.concatenate([d['scaf'][tri], d['blocks'][tri].ravel()]))
    bt = Batcher(d, idv, y, cut=bool(a.cut))
    model = Net(n_ids=len(idv) if a.ids else 0, **kw)
    print(f'fold {a.fold} train {len(tri)} val {len(vai)} nnew {np.bincount(nnew)} newscaf {ns.sum()}', flush=True)
    model, hist = train_model(model, bt, tri, a.epochs, lr=a.lr, wd=a.wd, bs=a.bs, seed=a.seed,
                              val=(bt, vai, y[vai], nnew), eval_every=a.eval_every, ls=a.ls,
                              log=lambda s: print(s, flush=True))
    p = predict(model, bt, vai)
    np.save(os.path.join(HERE, 'oof', f'{a.tag}_f{a.fold}.npy'), p.astype(np.float32))
    print('RESULT', json.dumps(dict(vars(a), res=hist[-1][1])), flush=True)
else:
    idv = np.unique(np.concatenate([d['scaf'], d['blocks'].ravel()]))
    bt = Batcher(d, idv, y, cut=bool(a.cut))
    model = Net(n_ids=len(idv) if a.ids else 0, **kw)
    model, _ = train_model(model, bt, np.arange(len(y)), a.epochs, lr=a.lr, wd=a.wd, bs=a.bs, seed=a.seed, ls=a.ls,
                           log=lambda s: print(s, flush=True))
    for k in ['public', 'private']:
        bk = Batcher(Fz[k], idv, None, cut=bool(a.cut))
        p = predict(model, bk, np.arange(len(Fz[k]['na'])))
        np.save(os.path.join(HERE, 'oof', f'{a.tag}_{k}_s{a.seed}.npy'), p.astype(np.float32))
    print('DONE', flush=True)
