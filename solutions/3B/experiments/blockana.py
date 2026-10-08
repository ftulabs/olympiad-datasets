import numpy as np, torch, pandas as pd, collections
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
torch.set_num_threads(1)
z=np.load('cache.npz'); T=['bind_BRD4','bind_HSA','bind_sEH']
y=pd.read_csv('/home/minh/Desktop/olympiad_ai/warmup/3B_molecule_binding/dataset/train/train.csv')[T].values.astype(np.float32)
bl=z['tr_blocks']; sc=z['tr_scaf']
ub,bi=np.unique(bl,return_inverse=True); bi=bi.reshape(bl.shape); us,si=np.unique(sc,return_inverse=True)
B=torch.from_numpy(bi); S=torch.from_numpy(si); Y=torch.from_numpy(y)
eb=torch.zeros(len(ub),3,requires_grad=True); es=torch.zeros(len(us),3,requires_grad=True); b0=torch.full((3,),-3.0,requires_grad=True)
opt=torch.optim.Adam([eb,es,b0],lr=0.05)
for it in range(500):
    l=eb[B].sum(1)+es[S]+b0
    loss=torch.nn.functional.binary_cross_entropy_with_logits(l,Y)+1e-4*(eb**2).sum()
    opt.zero_grad(); loss.backward(); opt.step()
w=eb.detach().numpy(); print('scaf eff',es.detach().numpy().round(2))
np.save('block_w.npy',w)
# block fingerprints from a representative molecule
ao=np.concatenate([[0],np.cumsum(z['tr_na'])]); part=z['tr_part']; keys=z['tr_keys']; x=z['tr_x']
rep={}
for i in range(len(y)):
    for k in range(3):
        if bi[i,k] not in rep: rep[bi[i,k]]=(i,k+1)
for R in [0,1,2]:
    rows=[]
    for b in range(len(ub)):
        i,p=rep[b]; sl=slice(ao[i],ao[i+1]); msk=part[sl]==p
        rows.append(collections.Counter(keys[:R+1,sl][:,msk].ravel().tolist()))
    vocab=sorted({k for r in rows for k in r}); vi={k:j for j,k in enumerate(vocab)}
    X=np.zeros((len(ub),len(vocab)))
    for b,r in enumerate(rows):
        for k,c in r.items(): X[b,vi[k]]=c
    for t in range(3):
        p=cross_val_predict(RidgeCV(alphas=np.logspace(-2,3,12)),X,w[:,t],cv=KFold(5,shuffle=True,random_state=0))
        print('R',R,'nfeat',X.shape[1],T[t],'CV corr',np.corrcoef(p,w[:,t])[0,1].round(3))
# atom feature counts (element,deg,nH,arom,ring...)
X2=[]
for b in range(len(ub)):
    i,p=rep[b]; sl=slice(ao[i],ao[i+1]); msk=part[sl]==p
    X2.append(x[sl][msk].sum(0))
X2=np.array(X2,float)
for t in range(3):
    p=cross_val_predict(RidgeCV(alphas=np.logspace(-2,3,12)),X2,w[:,t],cv=KFold(5,shuffle=True,random_state=0))
    print('atomfeat',T[t],np.corrcoef(p,w[:,t])[0,1].round(3))
