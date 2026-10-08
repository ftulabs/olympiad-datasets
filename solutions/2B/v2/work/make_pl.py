"""Fold-clean pseudo-labels for test images: for fold k, every component is fit without fold-k labels.
usage: python make_pl.py NAME expert1 expert2 ... (image experts = remote tags, '+' for seed avg)"""
import sys, glob, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from common import *
from experts_tt import text_fit_predict, tab_fit_predict, tab_matrix, typo_fixer
tr,te,y,folds,npu=load()
name, ex = sys.argv[1], sys.argv[2:]
clean=make_cleaner(pd.concat([tr.province,te.province]).unique())
T=tr.text.map(clean).values; Tt=te.text.map(clean).values
fx=typo_fixer(np.concatenate([T,Tt])); T=np.array([fx(t) for t in T]); Tt=np.array([fx(t) for t in Tt])
X=tab_matrix(tr,tr); Xt=tab_matrix(te,tr)
L=lambda p: np.log(np.clip(p,1e-4,1))
oof_tt={n:np.load(f'oof/{n}.npz')['oof'] for n in ['text','tab']}
def img(tag,k):
    oof=np.zeros((len(y),5)); 
    for t in tag.split('+'):
        for f in range(5):
            z=np.load(f'remote_out/{t}_f{f}.npz'); oof[z['idx']]+=z['oof']/len(tag.split('+'))
    tp=np.mean([np.load(f'remote_out/{t}_f{k}.npz')['test'] for t in tag.split('+')],0)
    return oof,tp
full=0
for k,(a,b) in enumerate(folds):
    comps_oof=[oof_tt['text'],oof_tt['tab']]; comps_te=[text_fit_predict(T[a],y[a],Tt),tab_fit_predict(X[a],y[a],Xt)]
    for t in ex:
        o,p=img(t,k); comps_oof.append(o); comps_te.append(p)
    Z=np.hstack([L(o) for o in comps_oof]); Zt=np.hstack([L(p) for p in comps_te])
    m=LogisticRegression(C=0.3,max_iter=5000).fit(Z[a],y[a])
    P=m.predict_proba(Zt); np.savez(f'pl_{name}_f{k}.npz',test=P)
    print(k,'val F1 of fold stack %.4f'%f1_score(y[b],m.predict(Z[b]),average='macro'),'conf>.9 %.3f'%(P.max(1)>.9).mean(),flush=True)
    full=full+P/5
np.savez(f'pl_{name}.npz',test=full)
