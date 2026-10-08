import pickle, numpy as np, pandas as pd, collections
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as aps
from scipy import sparse
D='/home/minh/Desktop/olympiad_ai/warmup/3B_molecule_binding/dataset/'
T=['bind_BRD4','bind_HSA','bind_sEH']
tr=pd.read_csv(D+'train/train.csv'); y=tr[T].values
F=pickle.load(open('feats.pkl','rb')); d=F['train']
B=d['blocks']; ub,inv=np.unique(B,return_inverse=True); inv=inv.reshape(B.shape)
us,sinv=np.unique(d['scaf'],return_inverse=True)
n=len(y)
for t in range(3):
    rate=np.zeros(len(ub)); cnt=np.zeros(len(ub))
    for k in range(3):
        np.add.at(rate,inv[:,k],y[:,t]); np.add.at(cnt,inv[:,k],1)
    r=rate/cnt
    print(T[t],'block rate pct',np.round(np.percentile(r,[0,10,25,50,75,90,95,99,100]),3))
    print('  top rates',np.round(np.sort(r)[-15:],3))
# pairs: does a molecule with 2 hot blocks have higher rate? additivity test via logistic regression on onehot blocks + scaffold
X=sparse.hstack([sparse.csr_matrix((np.ones(3*n),(np.repeat(np.arange(n),3),inv.ravel())),shape=(n,len(ub))),
                 sparse.csr_matrix((np.ones(n),(np.arange(n),sinv)),shape=(n,len(us)))]).tocsr()
rng=np.random.RandomState(0); perm=rng.permutation(n); va=perm[:8000]; trn=perm[8000:]
for t in range(3):
    for C in [0.1,1,10]:
        m=LogisticRegression(C=C,max_iter=2000).fit(X[trn],y[trn,t]); p=m.predict_proba(X[va])[:,1]
        print(T[t],'C',C,'random-split AP onehot-additive',round(aps(y[va,t],p),4))
