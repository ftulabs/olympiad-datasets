import pickle, numpy as np, pandas as pd, collections
D='/home/minh/Desktop/olympiad_ai/warmup/3B_molecule_binding/dataset/'
T=['bind_BRD4','bind_HSA','bind_sEH']
tr=pd.read_csv(D+'train/train.csv'); y=tr[T].values
F=pickle.load(open('feats.pkl','rb'))
d=F['train']
print('scaf==0 (decomp fail):', {k:(F[k]['scaf']==0).sum() for k in F})
for k in F:
    print(k,'n unique scaf',len(np.unique(F[k]['scaf'])),'blocks',len(np.unique(F[k]['blocks'])))
trs=set(d['scaf']); trb=set(d['blocks'].ravel())
for k in ['public','private']:
    s=F[k]['scaf']; b=F[k]['blocks']
    ns=np.array([x not in trs for x in s]); nb=np.array([[x not in trb for x in r] for r in b]).sum(1)
    print(k,'unseen scaf frac',ns.mean(),'unseen blocks count dist',np.bincount(nb,minlength=4)/len(nb))
    print(pd.crosstab(ns,nb))
# scaffolds counts
sc=collections.Counter(d['scaf']); print('scaf counts train',sc.most_common(20))
# label rate by scaffold
df=pd.DataFrame(y,columns=T); df['scaf']=d['scaf']
print(df.groupby('scaf')[T].agg(['mean','count']).sort_values(('bind_BRD4','count'),ascending=False).head(20))
# block counts
bc=collections.Counter(d['blocks'].ravel()); print('n blocks',len(bc),'count quantiles',np.percentile(list(bc.values()),[10,50,90,99]))
# id
print(np.corrcoef(tr.id,y.T)[0,1:])
for k,f in [('train',tr)]: print(tr.id.min(),tr.id.max())
