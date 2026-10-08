import sys, os, numpy as np
sys.path.insert(0,'.')
os.environ['CUDA_VISIBLE_DEVICES']=''
from lib2 import load, make_splits, eval_val
from common import fp_matrix
from scipy import sparse
from sklearn.linear_model import LogisticRegression
Fz,y=load(); d=Fz['train']; n=len(y)
os.makedirs('oof',exist_ok=True)
def idmat(dd, idv):
    H=np.concatenate([dd['scaf'][:,None],dd['blocks']],1); m=len(H)
    pos=np.clip(np.searchsorted(idv,H),0,len(idv)-1); ok=idv[pos]==H
    r=np.repeat(np.arange(m),4).reshape(m,4)
    return sparse.csr_matrix((np.ones(ok.sum()),(r[ok],pos[ok])),shape=(m,len(idv)))
def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
which=sys.argv[1] if len(sys.argv)>1 else 'both'
for f,(tri,vai,nnew,ns) in enumerate(make_splits(d)):
    if which in ('id','both'):
        idv=np.unique(np.concatenate([d['scaf'][tri],d['blocks'][tri].ravel()]))
        X=idmat(d,idv); P=np.zeros((len(vai),3))
        for t in range(3):
            P[:,t]=logit(LogisticRegression(C=0.5,max_iter=3000).fit(X[tri],y[tri,t]).predict_proba(X[vai])[:,1])
        np.save(f'oof/lrid_f{f}.npy',P.astype(np.float32)); print('lrid',f,eval_val(y[vai],P,nnew),flush=True)
    if which in ('fp','both'):
        K=np.unique(np.concatenate([np.unique(d['keys'][:4,d['ao'][i]:d['ao'][i+1]]) for i in tri]))
        X=fp_matrix(d,K,3); X.data=np.log1p(X.data); P=np.zeros((len(vai),3))
        for t in range(3):
            P[:,t]=logit(LogisticRegression(C=0.03,max_iter=3000).fit(X[tri],y[tri,t]).predict_proba(X[vai])[:,1])
        np.save(f'oof/lrfp_f{f}.npy',P.astype(np.float32)); print('lrfp',f,eval_val(y[vai],P,nnew),flush=True)
