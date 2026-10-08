import sys, os, pickle, numpy as np
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'.')
import torch; torch.set_num_threads(1)
from fp_explore import matrix, transform
from lib2 import load, torch_logreg
R=pickle.load(open('fpkeys.pkl','rb')); Fz,y=load()
spec=sys.argv[1]; C=float(sys.argv[2]); tag=sys.argv[3]
X,voc=matrix(R['train'],spec); TRF=os.environ.get('TR','log'); X=transform(X,TRF).tocsr()
Xs=[transform(matrix(R[k],spec,voc)[0],TRF).tocsr() for k in ['public','private']]
Pp,Pq=torch_logreg(X,y,Xs,C=C)
np.save(f'oof/{tag}_public_s0.npy',Pp.astype(np.float32)); np.save(f'oof/{tag}_private_s0.npy',Pq.astype(np.float32)); print('done')
