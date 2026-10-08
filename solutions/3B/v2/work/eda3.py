import pickle, numpy as np, pandas as pd, collections
from scipy.stats import spearmanr
D='/home/minh/Desktop/olympiad_ai/warmup/3B_molecule_binding/dataset/'
T=['bind_BRD4','bind_HSA','bind_sEH']
tr=pd.read_csv(D+'train/train.csv'); y=tr[T].values
F=pickle.load(open('feats.pkl','rb')); d=F['train']
B=d['blocks']; ub,inv=np.unique(B,return_inverse=True); inv=inv.reshape(B.shape)
cnt=np.bincount(inv.ravel())
anyy=y.max(1)
for t in range(4):
    yy=anyy if t==3 else y[:,t]
    r=np.bincount(inv.ravel(),weights=np.repeat(yy,3))/cnt
    print(t,'spearman count vs rate',spearmanr(cnt,r))
# count distribution
print(np.sort(cnt)[:10],np.sort(cnt)[-10:])
# test block frequencies
allb=collections.Counter(np.concatenate([F[k]['blocks'].ravel() for k in F]))
pu=collections.Counter(F['public']['blocks'].ravel()); pr=collections.Counter(F['private']['blocks'].ravel())
trb=set(ub)
newpu=[c for b,c in pu.items() if b not in trb]; newpr=[c for b,c in pr.items() if b not in trb]
print('new blocks public',len(newpu),'counts pct',np.percentile(newpu,[0,25,50,75,100]))
print('new blocks private',len(newpr),'counts pct',np.percentile(newpr,[0,25,50,75,100]))
print('overlap of new blocks pub/priv', len(set(b for b in pu if b not in trb)&set(b for b in pr if b not in trb)))
print('seen blocks in public counts pct',np.percentile([c for b,c in pu.items() if b in trb],[0,25,50,75,100]))
# blocks per position: are blocks partitioned into roles? co-occurrence: each block appears with which "slot"? use atom count
# block size
na=np.array([0])
# multiplicity: same block twice in a molecule?
print('dup blocks in mol', np.mean([len(set(r))<3 for r in B]))
# n_atoms
print('mean rate by number of hot...')
