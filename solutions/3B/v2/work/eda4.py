import pickle, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import average_precision_score as aps
from scipy import sparse
D='/home/minh/Desktop/olympiad_ai/warmup/3B_molecule_binding/dataset/'
T=['bind_BRD4','bind_HSA','bind_sEH']
tr=pd.read_csv(D+'train/train.csv'); y=tr[T].values
F=pickle.load(open('feats.pkl','rb')); d=F['train']
B=d['blocks']; ub,inv=np.unique(B,return_inverse=True); inv=inv.reshape(B.shape)
us,sinv=np.unique(d['scaf'],return_inverse=True); n=len(y)
X=sparse.hstack([sparse.csr_matrix((np.ones(3*n),(np.repeat(np.arange(n),3),inv.ravel())),shape=(n,len(ub))),
                 sparse.csr_matrix((np.ones(n),(np.arange(n),sinv)),shape=(n,len(us)))]).tocsr()
rng=np.random.RandomState(0); perm=rng.permutation(n); va=perm[:8000]; trn=perm[8000:]
# OOF coefs within trn
W=np.zeros((n,3,3)) # mol, target, slot
folds=np.array_split(rng.permutation(trn),5)
coefs_full=[]
for t in range(3):
    m=LogisticRegression(C=1,max_iter=3000).fit(X[trn],y[trn,t]); coefs_full.append(m.coef_[0][:len(ub)])
    W[va,t]=m.coef_[0][inv[va]]
    for f in folds:
        rest=np.setdiff1d(trn,f); mm=LogisticRegression(C=1,max_iter=3000).fit(X[rest],y[rest,t]); W[f,t]=mm.coef_[0][inv[f]]
Ws=np.sort(W,2).reshape(n,9)
feat=np.concatenate([Ws,sinv[:,None]],1)
for t in range(3):
    base=W[va,t].sum(1)
    g=HGB(max_iter=300,learning_rate=0.05,max_leaf_nodes=15,categorical_features=[9]).fit(feat[trn],y[trn,t])
    print(T[t],'additive',round(aps(y[va,t],base),4),'HGB on sorted block coefs(all targets)',round(aps(y[va,t],g.predict_proba(feat[va])[:,1]),4))
    # max-based
    print('   max',round(aps(y[va,t],W[va,t].max(1)),4))
