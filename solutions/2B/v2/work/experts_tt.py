"""text + tab experts: OOF (v1 folds) + full-train test preds -> oof/text.npz, oof/tab.npz"""
import numpy as np, pandas as pd, scipy.sparse as sp
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, log_loss
from common import *

def typo_fixer(texts):
    cnt=Counter(w for t in texts for w in t.split()); good=[w for w,c in cnt.items() if c>=10]
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
    return lambda t:' '.join(fix.get(w,w) for w in t.split())

def text_fit_predict(A,y,B,C=2.0):
    vw=TfidfVectorizer(ngram_range=(1,3),min_df=2,sublinear_tf=True); vc=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=3,sublinear_tf=True)
    XA=sp.hstack([vw.fit_transform(A),vc.fit_transform(A)]).tocsr(); XB=sp.hstack([vw.transform(B),vc.transform(B)]).tocsr()
    return LogisticRegression(C=C,max_iter=3000).fit(XA,y).predict_proba(XB)

def tab_matrix(df,ref):
    X=df[NUM+CAT].copy(); X['n_missing']=df.nitrogen_kg_ha.isna().astype(int); X['nitrogen_kg_ha']=X.nitrogen_kg_ha.fillna(-1)
    X['trange']=df.temp_max_7d-df.temp_min_7d
    for c in CAT: X[c]=pd.Categorical(df[c],categories=sorted(ref[c].unique())).codes
    X['n_per_day']=df.nitrogen_kg_ha.fillna(-1)/(df.days_after_sowing+1); X['rain_storm']=df.rainfall_7d_mm*(1+df.storm_last_3d); X['hum_tmin']=df.humidity_7d-3*df.temp_min_7d
    return X.values.astype(float)
CM=[False]*8+[True]*3+[False]*5
def tab_fit_predict(A,y,B):
    return HistGradientBoostingClassifier(learning_rate=0.01,max_iter=800,max_leaf_nodes=4,min_samples_leaf=30,l2_regularization=1.0,categorical_features=CM,random_state=0).fit(A,y).predict_proba(B)

if __name__=='__main__':
    tr,te,y,folds,npu=load()
    clean=make_cleaner(pd.concat([tr.province,te.province]).unique())
    T=tr.text.map(clean).values; Tt=te.text.map(clean).values
    fx=typo_fixer(np.concatenate([T,Tt])); T=np.array([fx(t) for t in T]); Tt=np.array([fx(t) for t in Tt])
    X=tab_matrix(tr,tr); Xt=tab_matrix(te,tr)
    for name,fn,A,B in [('text',text_fit_predict,T,Tt),('tab',tab_fit_predict,X,Xt)]:
        oof=np.zeros((len(y),5))
        for a,b in folds: oof[b]=fn(A[a],y[a],A[b])
        print(name,'F1=%.4f ll=%.4f'%(f1_score(y,oof.argmax(1),average='macro'),log_loss(y,oof)),flush=True)
        np.savez(f'oof/{name}.npz',oof=oof,test=fn(A,y,B))
