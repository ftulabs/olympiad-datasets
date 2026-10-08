import numpy as np, sys
from common import *
from sklearn.linear_model import LogisticRegression
F,y=load(); d=F['train']
spl=make_splits(d)
for f,(tri,vai,nnew,ns) in enumerate(spl[:2]):
    print('fold',f,'train',len(tri),'val',len(vai),'nnew',np.bincount(nnew),'newscaf',ns.sum())
for radius in [1,2,3]:
    for C in [0.03,0.1,0.3]:
        res=[]
        for f,(tri,vai,nnew,ns) in enumerate(spl[:2]):
            K=np.unique(np.concatenate([np.unique(d['keys'][:radius+1,d['ao'][i]:d['ao'][i+1]]) for i in tri[:20000]]))
            X=fp_matrix(d,K,radius); X.data=np.log1p(X.data)
            P=np.zeros((len(vai),3))
            for t in range(3):
                m=LogisticRegression(C=C,max_iter=2000).fit(X[tri],y[tri,t]); P[:,t]=m.predict_proba(X[vai])[:,1]
            res.append((mean_ap(y[vai],P), mean_ap(y[vai][nnew>=2],P[nnew>=2])))
        print('radius',radius,'C',C,np.round(np.mean(res,0),4),flush=True)
