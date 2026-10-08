"""EmbeddingBag MLP over log-count ECFP keys. usage: fpmlp.py spec epochs tag [fold|-1]"""
import sys, os, pickle, numpy as np, math
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'.')
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get('NT','2')))
from fp_explore import matrix, transform
from lib2 import load, make_splits, eval_val
R=pickle.load(open('fpkeys.pkl','rb')); Fz,y=load(); d=Fz['train']
spec,ep,tag=sys.argv[1],int(sys.argv[2]),sys.argv[3]
X,voc=matrix(R['train'],spec); X=transform(X,'log').tocsr()
class M(nn.Module):
    def __init__(s,V,h=256,dr=0.5):
        super().__init__(); s.e=nn.EmbeddingBag(V,h,mode='sum'); s.b=nn.Parameter(torch.zeros(h))
        s.f=nn.Sequential(nn.ReLU(),nn.Dropout(dr),nn.Linear(h,h),nn.ReLU(),nn.Dropout(dr),nn.Linear(h,3))
        nn.init.normal_(s.e.weight,std=0.02)
    def forward(s,X):
        X=X.tocsr(); idx=torch.from_numpy(X.indices.astype(np.int64)); off=torch.from_numpy(X.indptr[:-1].astype(np.int64))
        w=torch.from_numpy(X.data.astype(np.float32)); return s.f(s.e(idx,off,per_sample_weights=w)+s.b)
def fit(tri, seed=0, bs=256, lr=2e-3, wd=1e-2):
    torch.manual_seed(seed); rng=np.random.RandomState(seed); m=M(X.shape[1])
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd); steps=ep*math.ceil(len(tri)/bs)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,total_steps=steps,pct_start=0.1)
    Y=torch.from_numpy(y)
    for e in range(ep):
        m.train(); p=rng.permutation(tri)
        for s in range(0,len(p),bs):
            ii=p[s:s+bs]; loss=F.binary_cross_entropy_with_logits(m(X[ii]),Y[ii]); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval(); return m
fold=int(sys.argv[4]) if len(sys.argv)>4 else None
if fold is None or fold>=0:
    res=[]
    for f,(tri,vai,nnew,ns) in enumerate(make_splits(d)):
        if fold is not None and f!=fold: continue
        m=fit(tri)
        with torch.no_grad(): P=m(X[vai]).numpy()
        np.save(f'oof/{tag}_f{f}.npy',P.astype(np.float32)); r=eval_val(y[vai],P,nnew); res.append([r['all'],r['mix']]); print(f,np.round(res[-1],4),flush=True)
    print(tag,'mean',np.round(np.mean(res,0),4))
else:
    for s in range(3):
        m=fit(np.arange(len(y)),seed=s)
        for k in ['public','private']:
            Xt=transform(matrix(R[k],spec,voc)[0],'log').tocsr()
            with torch.no_grad(): np.save(f'oof/{tag}_{k}_s{s}.npy',m(Xt).numpy().astype(np.float32))
    print('done')
