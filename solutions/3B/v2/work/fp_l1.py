import sys, os, pickle, numpy as np
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'.')
from fp_explore import matrix, transform, logit
from lib2 import load, make_splits, eval_val
from sklearn.linear_model import LogisticRegression
recs=pickle.load(open('fpkeys.pkl','rb'))['train']; Fz,y=load(); d=Fz['train']
X,_=matrix(recs,sys.argv[1]); X=transform(X,os.environ.get('TR','log')).tocsr()
for C in [float(c) for c in sys.argv[2:]]:
    res=[]
    for f,(tri,vai,nnew,ns) in enumerate(make_splits(d)):
        P=np.zeros((len(vai),3)); nz=[]
        for t in range(3):
            m=LogisticRegression(C=C,penalty='l1',solver='liblinear',max_iter=2000).fit(X[tri],y[tri,t]); P[:,t]=m.decision_function(X[vai]); nz.append((m.coef_!=0).sum())
        r=eval_val(y[vai],P,nnew); res.append([r['all'],r['mix'],r['s0'],r['s3']])
        print(f, np.round(res[-1],4), nz, flush=True)
    print(sys.argv[1],'L1 C',C,np.round(np.mean(res,0),4),flush=True)
