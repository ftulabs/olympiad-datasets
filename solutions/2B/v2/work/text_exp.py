import sys, numpy as np, scipy.sparse as sp, pandas as pd, re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from common import *
tr,te,y,folds,npu=load()
clean=make_cleaner(pd.concat([tr.province,te.province]).unique())
T=tr.text.map(clean).values; Tt=te.text.map(clean).values
# typo normaliser: map rare tokens to the most frequent token within edit distance 1 (dictionary from train+test tokens)
cnt=Counter(w for t in np.concatenate([T,Tt]) for w in t.split())
good=[w for w,c in cnt.items() if c>=10]
def ed1(a,b):
    if abs(len(a)-len(b))>1: return False
    if len(a)==len(b):
        d=[i for i in range(len(a)) if a[i]!=b[i]]
        return len(d)==1 or (len(d)==2 and d[1]==d[0]+1 and a[d[0]]==b[d[1]] and a[d[1]]==b[d[0]])
    if len(a)>len(b): a,b=b,a
    return any(b[:i]+b[i+1:]==a for i in range(len(b)))
fix={}
for w,c in cnt.items():
    if c<10 and len(w)>=2:
        cands=[g for g in good if ed1(w,g)]
        if cands: fix[w]=max(cands,key=lambda g:cnt[g])
print('fix map size',len(fix), list(fix.items())[:15])
norm=lambda t:' '.join(fix.get(w,w) for w in t.split())
TN=np.array([norm(t) for t in T]); TtN=np.array([norm(t) for t in Tt])
def run(name, feats, C=2.0):
    oof=np.zeros((len(y),5))
    for a,b in folds:
        mats=[f() for f in feats]
        A=sp.hstack([m.fit_transform(X[a]) for m,X in zip(mats,[f.X for f in feats])]).tocsr()
        B=sp.hstack([m.transform(X[b]) for m,X in zip(mats,[f.X for f in feats])]).tocsr()
        oof[b]=LogisticRegression(C=C,max_iter=3000).fit(A,y[a]).predict_proba(B)
    print(f'{name:40s} F1={f1_score(y,oof.argmax(1),average="macro"):.4f} ll={log_loss(y,oof):.4f}',flush=True)
    return oof
def F(fn,X):
    f=fn; f.X=X; return f
w13=lambda: TfidfVectorizer(ngram_range=(1,3),min_df=2,sublinear_tf=True)
c25=lambda: TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=3,sublinear_tf=True)
run('v1 w13+c25', [F(w13,T),F(c25,T)])
run('norm w13+c25', [F(w13,TN),F(c25,TN)])
for C in [0.5,1,4]: run(f'norm w13+c25 C={C}', [F(w13,TN),F(c25,TN)],C)
run('norm w14 only', [F(lambda: TfidfVectorizer(ngram_range=(1,4),min_df=2,sublinear_tf=True),TN)])
run('norm w13+c16', [F(w13,TN),F(lambda: TfidfVectorizer(analyzer='char_wb',ngram_range=(1,6),min_df=3,sublinear_tf=True),TN)])
run('norm binary w13', [F(lambda: TfidfVectorizer(ngram_range=(1,3),min_df=2,binary=True,use_idf=False,norm=None),TN)],0.1)
