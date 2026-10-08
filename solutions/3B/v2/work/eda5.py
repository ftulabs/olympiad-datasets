import pickle, numpy as np, pandas as pd
F=pickle.load(open('feats.pkl','rb')); d=F['train']
B=d['blocks']; ub,inv=np.unique(B,return_inverse=True); inv=inv.reshape(B.shape); nb=len(ub)
C=np.zeros((nb,nb))
for a,b in [(0,1),(0,2),(1,2)]:
    np.add.at(C,(inv[:,a],inv[:,b]),1); np.add.at(C,(inv[:,b],inv[:,a]),1)
partners=(C>0).sum(1)
print('distinct partners pct',np.percentile(partners,[0,10,50,90,100]))
w,v=np.linalg.eigh(C); print('top eig',np.round(w[-4:],1),'bottom',np.round(w[:4],1))
sgn=np.sign(v[:,0]); print('bottom eigvec sign split',np.bincount((sgn>0).astype(int)))
# blocks per scaffold
us,sinv=np.unique(d['scaf'],return_inverse=True)
M=np.zeros((nb,len(us)))
for k in range(3): np.add.at(M,(inv[:,k],sinv),1)
print('blocks by scaffold presence', np.bincount((M>0).sum(1)))
print(M[:10])
# block atom sizes
sizes={}
part=d['part']; ao=d['ao']
for i in range(2000):
    p=part[ao[i]:ao[i+1]]
    for k in range(3): sizes.setdefault(B[i,k],(p==k+1).sum())
print('block size pct',np.percentile(list(sizes.values()),[0,10,50,90,100]))
sc=[ (part[ao[i]:ao[i+1]]==0).sum() for i in range(2000)]
print('scaffold size by scaf', pd.Series(sc).groupby(sinv[:2000]).describe())
