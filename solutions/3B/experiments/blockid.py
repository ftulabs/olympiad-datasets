import numpy as np, torch, pandas as pd
from sklearn.metrics import average_precision_score
torch.set_num_threads(1)
z=np.load('cache.npz'); T=['bind_BRD4','bind_HSA','bind_sEH']
y=pd.read_csv('/home/minh/Desktop/olympiad_ai/warmup/3B_molecule_binding/dataset/train/train.csv')[T].values.astype(np.float32)
bl=z['tr_blocks']; sc=z['tr_scaf']
ub,bi=np.unique(bl,return_inverse=True); bi=bi.reshape(bl.shape); us,si=np.unique(sc,return_inverse=True)
rng=np.random.RandomState(0); perm=rng.permutation(len(y)); va=perm[:8000]; tr=perm[8000:]
B=torch.from_numpy(bi); S=torch.from_numpy(si); Y=torch.from_numpy(y)
for wd in [1e-4,1e-3]:
    eb=torch.zeros(len(ub),3,requires_grad=True); es=torch.zeros(len(us),3,requires_grad=True); b0=torch.full((3,),-3.0,requires_grad=True)
    opt=torch.optim.Adam([eb,es,b0],lr=0.05)
    for it in range(400):
        l=eb[B[tr]].sum(1)+es[S[tr]]+b0
        loss=torch.nn.functional.binary_cross_entropy_with_logits(l,Y[tr])+wd*(eb**2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    p=(eb[B[va]].sum(1)+es[S[va]]+b0).detach().numpy()
    aps=[average_precision_score(y[va,k],p[:,k]) for k in range(3)]
    print('additive blockID wd',wd,np.round(aps,4),np.mean(aps))
    # inspect learned block weight spread
    w=eb.detach().numpy(); print(' block w quantiles',np.round(np.quantile(w,[0,.5,.9,.99,1],axis=0),2).T)
