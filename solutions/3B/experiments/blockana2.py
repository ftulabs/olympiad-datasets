import numpy as np, pandas as pd, collections, sys
sys.path.insert(0,'.')
from feats import *
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_predict, KFold
z=np.load('cache.npz'); w=np.load('block_w.npy')
D='/home/minh/Desktop/olympiad_ai/warmup/3B_molecule_binding/dataset/'
tr=pd.read_csv(D+'train/train.csv')
ub,bi=np.unique(z['tr_blocks'],return_inverse=True); bi=bi.reshape(-1,3)
part=z['tr_part']; ao=np.concatenate([[0],np.cumsum(z['tr_na'])])
rep={}
for i in range(len(bi)):
    for k in range(3):
        if bi[i,k] not in rep: rep[bi[i,k]]=(i,k+1)
# isolated block subgraph ECFP
def iso_keys(i,p,R):
    m=mol_info(tr.atoms[i],tr.bonds[i]); mem=np.where(part[ao[i]:ao[i+1]]==p)[0]; S=set(mem.tolist())
    # attachment atom: atom in block bonded to outside
    att={a for a in S for j,t in m['adj'][a] if j not in S}
    h={a:h64(f"{m['at'][a]}|{m['deg'][a]}|{m['nh'][a]}|{m['arom'][a]}|{m['inring'][a]}|{m['rs'][a]}|{a in att}") for a in S}
    keys=list(h.values())
    for r in range(1,R+1):
        h={a:h64(f"{r}|{h[a]}|"+str(sorted((t,h[j]) for j,t in m['adj'][a] if j in S))) for a in S}
        keys+=list(h.values())
    return collections.Counter(keys)
def mat(rows):
    vocab=sorted({k for r in rows for k in r}); vi={k:j for j,k in enumerate(vocab)}
    X=np.zeros((len(rows),len(vocab)))
    for b,r in enumerate(rows):
        for k,c in r.items(): X[b,vi[k]]=c
    return X
def tanimoto(A,B):
    num=np.minimum(A[:,None,:],B[None,:,:]).sum(-1); den=np.maximum(A[:,None,:],B[None,:,:]).sum(-1)
    return num/np.maximum(den,1e-9)
cv=KFold(5,shuffle=True,random_state=0)
for R in [1,2,3]:
    X=mat([iso_keys(*rep[b],R) for b in range(len(ub))])
    K=tanimoto(X,X)
    for name in ['ridge','rf','tani']:
        cs=[]
        for t in range(3):
            if name=='ridge': p=cross_val_predict(RidgeCV(alphas=np.logspace(-2,3,12)),X,w[:,t],cv=cv)
            elif name=='rf': p=cross_val_predict(RandomForestRegressor(300,min_samples_leaf=2,max_features=0.3,n_jobs=2,random_state=0),X,w[:,t],cv=cv)
            else:
                p=np.zeros(len(ub))
                for a,bb in cv.split(X):
                    kr=KernelRidge(alpha=0.3,kernel='precomputed').fit(K[np.ix_(a,a)],w[a,t]-w[a,t].mean()); p[bb]=kr.predict(K[np.ix_(bb,a)])+w[a,t].mean()
            cs.append(np.corrcoef(p,w[:,t])[0,1])
        print('iso R',R,X.shape[1],name,np.round(cs,3),flush=True)
