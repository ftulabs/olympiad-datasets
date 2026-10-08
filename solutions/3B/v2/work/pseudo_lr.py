import sys, os, pickle, numpy as np
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'.')
import torch; torch.set_num_threads(1)
from scipy import sparse
from fp_explore import matrix, transform
from lib2 import load, make_splits, eval_val, torch_logreg
R=pickle.load(open('fpkeys.pkl','rb'))['train']; Fz,y=load(); d=Fz['train']
X,_=matrix(R,'ef3'); X=transform(X,'log').tocsr()
def zs(p): return (p-p.mean(0))/(p.std(0)+1e-9)
res=[]
for f,(tri,vai,nnew,ns) in enumerate(make_splits(d)):
    b=np.load(f'oof/base_f{f}.npy'); l=np.load(f'oof/lrfp1_f{f}.npy')
    teach=1/(1+np.exp(-(0.5*b+0.5*l)))
    Xa=sparse.vstack([X[tri],X[vai]]).tocsr(); ya=np.concatenate([y[tri],teach]).astype(np.float32)
    P,=torch_logreg(Xa,ya,[X[vai]],C=0.1)
    r0=eval_val(y[vai],l,nnew); r1=eval_val(y[vai],P,nnew)
    rb0=eval_val(y[vai],zs(b)*2+zs(l),nnew); rb1=eval_val(y[vai],zs(b)*2+zs(P),nnew)
    res.append([r0['mix'],r1['mix'],rb0['mix'],rb1['mix']]); print(f,np.round(res[-1],4),flush=True)
print('lr, lr+pseudo, blend, blend w/ pseudo-lr', np.round(np.mean(res,0),4))
