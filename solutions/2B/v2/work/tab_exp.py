import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score, log_loss
from common import *
tr,te,y,folds,npu=load()
def tabm(df, extra=False):
    X=df[NUM+CAT].copy(); X['n_missing']=df.nitrogen_kg_ha.isna().astype(int); X['nitrogen_kg_ha']=X.nitrogen_kg_ha.fillna(-1)
    X['trange']=df.temp_max_7d-df.temp_min_7d
    for c in CAT: X[c]=pd.Categorical(df[c],categories=sorted(tr[c].unique())).codes
    if extra:
        X['n_per_day']=df.nitrogen_kg_ha.fillna(-1)/(df.days_after_sowing+1)
        X['rain_storm']=df.rainfall_7d_mm*(1+df.storm_last_3d)
        X['hum_tmin']=df.humidity_7d-3*df.temp_min_7d
    return X.values.astype(float)
def run(name,mk,X):
    oof=np.zeros((len(y),5))
    for a,b in folds: oof[b]=mk().fit(X[a],y[a]).predict_proba(X[b])
    print(f'{name:40s} F1={f1_score(y,oof.argmax(1),average="macro"):.4f} ll={log_loss(y,oof):.4f}',flush=True); return oof
X=tabm(tr); Xe=tabm(tr,True)
cm=[False]*8+[True]*3+[False]*2
hgb=lambda lr=0.02,it=400,leaf=6,msl=30,cm=cm: (lambda: HistGradientBoostingClassifier(learning_rate=lr,max_iter=it,max_leaf_nodes=leaf,min_samples_leaf=msl,l2_regularization=1.0,categorical_features=cm,random_state=0))
run('v1 hgb',hgb(),X)
run('hgb lr.01 it800 leaf4',hgb(0.01,800,4),X)
run('hgb lr.03 it300 leaf8 msl50',hgb(0.03,300,8,50),X)
run('hgb extra',hgb(cm=cm+[False]*3),Xe)
# LR with splines-ish: one-hot cats + binned numerics
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
ct=lambda: make_pipeline(ColumnTransformer([('b',KBinsDiscretizer(8,encode='onehot',strategy='quantile'),list(range(8))+[12]),('o',OneHotEncoder(handle_unknown='ignore'),[8,9,10,11])]),LogisticRegression(C=0.3,max_iter=3000))
Xb=X.copy()
run('LR binned onehot',ct,Xb)
