import sys, os, pickle, numpy as np, time
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'.')
import torch; torch.set_num_threads(int(os.environ.get('NT','1')))
from fp_explore import matrix, transform
from lib2 import load, make_splits, eval_val, torch_logreg
recs=pickle.load(open('fpkeys.pkl','rb'))['train']; Fz,y=load(); d=Fz['train']
spec=sys.argv[1]; C=float(sys.argv[2]); save=sys.argv[3] if len(sys.argv)>3 else ''
X,_=matrix(recs,spec); X=transform(X,'log').tocsr()
res=[]
for f,(tri,vai,nnew,ns) in enumerate(make_splits(d)):
    t0=time.time(); P,=torch_logreg(X[tri],y[tri],[X[vai]],C=C)
    r=eval_val(y[vai],P,nnew); res.append([r['all'],r['mix']]); print(f,np.round(res[-1],4),f'{time.time()-t0:.0f}s',flush=True)
    if save: np.save(f'oof/{save}_f{f}.npy',P.astype(np.float32))
print(spec,C,'torchLR mean',np.round(np.mean(res,0),4))
