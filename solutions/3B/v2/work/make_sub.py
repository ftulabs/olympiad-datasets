"""compose submissions from saved full-fit test logits. usage: make_sub.py outdir tag:w [tag:w ...]  (tag = file prefix)"""
import sys, os, glob, numpy as np, pandas as pd
D='/home/minh/Desktop/olympiad_ai/warmup/3B_molecule_binding/dataset/'
T=['bind_BRD4','bind_HSA','bind_sEH']
def zs(p): return (p-p.mean(0))/(p.std(0)+1e-9)
out=sys.argv[1]; os.makedirs(out,exist_ok=True)
for k,f in [('public','public_test/public_test.csv'),('private','private_test/private_test.csv')]:
    df=pd.read_csv(D+f); z=0; W=0
    for spec in sys.argv[2:]:
        tag,w=spec.split(':'); fs=sorted(glob.glob(f'oof/{tag}_{k}_s*.npy'))
        p=np.mean([np.load(x) for x in fs],0); z=z+float(w)*zs(p); W+=float(w); print(k,tag,len(fs),'files')
    z=z/W; sub=pd.DataFrame(1/(1+np.exp(-z)),columns=T); sub.insert(0,'id',df['id'].values)
    sub.to_csv(f'{out}/{k}_submission.csv',index=False)
