import pickle, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as aps
from scipy import sparse
D='/home/minh/Desktop/olympiad_ai/warmup/3B_molecule_binding/dataset/'
T=['bind_BRD4','bind_HSA','bind_sEH']
tr=pd.read_csv(D+'train/train.csv'); y=tr[T].values
F=pickle.load(open('feats.pkl','rb')); d=F['train']
B=d['blocks']; ub,inv=np.unique(B,return_inverse=True); inv=inv.reshape(B.shape); nb=len(ub)
us,sinv=np.unique(d['scaf'],return_inverse=True); n=len(y)
rows=np.repeat(np.arange(n),3)
Xb=sparse.csr_matrix((np.ones(3*n),(rows,inv.ravel())),shape=(n,nb))
Xs=sparse.csr_matrix((np.ones(n),(np.arange(n),sinv)),shape=(n,len(us)))
Xbs=sparse.csr_matrix((np.ones(3*n),(rows,(inv*5+sinv[:,None]).ravel())),shape=(n,nb*5))
# pairwise block interactions hashed
pairs=[]
for a,b in [(0,1),(0,2),(1,2)]:
    lo=np.minimum(inv[:,a],inv[:,b]); hi=np.maximum(inv[:,a],inv[:,b]); pairs.append(lo*nb+hi)
pairs=np.stack(pairs,1); up,pinv=np.unique(pairs,return_inverse=True)
Xp=sparse.csr_matrix((np.ones(3*n),(rows,pinv.ravel())),shape=(n,len(up)))
rng=np.random.RandomState(0); perm=rng.permutation(n); va=perm[:8000]; trn=perm[8000:]
for name,X in [('b+s',sparse.hstack([Xb,Xs])),('b+s+bxs',sparse.hstack([Xb,Xs,Xbs])),('b+s+pairs',sparse.hstack([Xb,Xs,Xp]))]:
    X=X.tocsr(); r=[]
    for t in range(3):
        m=LogisticRegression(C=0.5,max_iter=3000).fit(X[trn],y[trn,t]); r.append(aps(y[va,t],m.predict_proba(X[va])[:,1]))
    print(name,np.round(r,4),round(np.mean(r),4))
