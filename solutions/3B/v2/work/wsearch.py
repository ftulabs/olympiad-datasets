import sys, os, numpy as np, itertools
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'.')
from stack import get, zs, SPL, y, eval_val
tags=sys.argv[1:]
folds=[f for f in range(6) if all(get(t,f) is not None for t in tags)]
Z={t:{f:zs(get(t,f)) for f in folds} for t in tags}
def score(w):
    rs=[eval_val(y[SPL[f][1]], sum(w[i]*Z[t][f] for i,t in enumerate(tags)), SPL[f][2]) for f in folds]
    return np.mean([r['mix'] for r in rs]), np.mean([r['all'] for r in rs])
print('folds',folds)
for t in tags: print(t, np.round(score([1.0 if u==t else 0 for u in tags]),4))
w=np.ones(len(tags)); best=score(w); print('equal',np.round(best,4))
grid=[0,0.25,0.5,0.75,1,1.5,2]
for it in range(2):
    for i in range(len(tags)):
        for g in grid:
            w2=w.copy(); w2[i]=g
            if w2.sum()==0: continue
            s=score(w2)
            if s[0]>best[0]+1e-4: best, w = s, w2
    print('iter',it,dict(zip(tags,w)),np.round(best,4))
