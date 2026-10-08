import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, '.')
import v1sol as S
D='/home/minh/Desktop/olympiad_ai/warmup/3B_molecule_binding/dataset/'
tr=pd.read_csv(D+'train/train.csv'); pu=pd.read_csv(D+'public_test/public_test.csv'); pr=pd.read_csv(D+'private_test/private_test.csv')
out={}
for k,df in [('train',tr),('public',pu),('private',pr)]:
    out[k]=S.featurize(df,1)
pickle.dump(out,open('feats.pkl','wb'))
