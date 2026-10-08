import pickle, numpy as np, pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score as aps
D='/home/minh/Desktop/olympiad_ai/warmup/3B_molecule_binding/dataset/'
T=['bind_BRD4','bind_HSA','bind_sEH']
def load():
    tr=pd.read_csv(D+'train/train.csv'); y=tr[T].values.astype(np.float32)
    F=pickle.load(open(__import__('os').path.dirname(__file__)+'/feats.pkl','rb'))
    return F,y
def mean_ap(y,p): return float(np.mean([aps(y[:,k],p[:,k]) for k in range(y.shape[1])]))
def make_splits(d, n_folds=4, block_frac=0.25, seed=0):
    """hold out a random fraction of blocks (+one scaffold per fold); val = molecules with any held-out part."""
    rng=np.random.RandomState(seed)
    blocks=np.unique(d['blocks']); scafs=np.unique(d['scaf']); rng.shuffle(blocks); rng.shuffle(scafs)
    nb=int(len(blocks)*block_frac); out=[]
    for f in range(n_folds):
        hb=blocks[(f*nb)%len(blocks):(f*nb)%len(blocks)+nb]
        nnew=np.isin(d['blocks'],hb).sum(1)
        ns=(d['scaf']==scafs[f%len(scafs)])
        tri=np.where((nnew==0)&~ns)[0]; vai=np.where((nnew>0)|ns)[0]
        out.append((tri,vai,nnew[vai],ns[vai]))
    return out
def fp_matrix(d, vocab, radius=2, counts=True):
    K=d['keys'][:radius+1]  # (r+1, atoms)
    mol=np.repeat(np.arange(len(d['na'])),d['na'])
    rows=np.tile(mol,radius+1); cols=K.ravel()
    pos=np.clip(np.searchsorted(vocab,cols),0,len(vocab)-1); ok=vocab[pos]==cols
    X=sparse.csr_matrix((np.ones(ok.sum()),(rows[ok],pos[ok])),shape=(len(d['na']),len(vocab)))
    X.sum_duplicates()
    if not counts: X.data[:]=1
    return X
