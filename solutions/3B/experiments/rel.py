import numpy as np, torch, pandas as pd
torch.set_num_threads(1)
z=np.load('cache.npz'); T=['bind_BRD4','bind_HSA','bind_sEH']
y=pd.read_csv('/home/minh/Desktop/olympiad_ai/warmup/3B_molecule_binding/dataset/train/train.csv')[T].values.astype(np.float32)
ub,bi=np.unique(z['tr_blocks'],return_inverse=True); bi=bi.reshape(-1,3); us,si=np.unique(z['tr_scaf'],return_inverse=True)
def fit(idx):
    B=torch.from_numpy(bi[idx]); S=torch.from_numpy(si[idx]); Y=torch.from_numpy(y[idx])
    eb=torch.zeros(len(ub),3,requires_grad=True); es=torch.zeros(len(us),3,requires_grad=True); b0=torch.full((3,),-3.0,requires_grad=True)
    opt=torch.optim.Adam([eb,es,b0],lr=0.05)
    for it in range(400):
        l=eb[B].sum(1)+es[S]+b0
        loss=torch.nn.functional.binary_cross_entropy_with_logits(l,Y)+1e-4*(eb**2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return eb.detach().numpy()
rng=np.random.RandomState(1); p=rng.permutation(len(y)); a=fit(p[:20000]); b=fit(p[20000:])
for t in range(3): print(T[t],'split-half corr of block effects',np.corrcoef(a[:,t],b[:,t])[0,1].round(3))
# interaction check: does block pair matter? compare additive vs pair-feature residual on random split
