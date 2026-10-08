import pandas as pd, numpy as np
from PIL import Image
D='/home/minh/Desktop/olympiad_ai/warmup/2B_rice_multimodal/dataset'
def stats(split):
  df=pd.read_csv(f'{D}/{split}/{split}.csv'); r=[]
  for p in df.image:
    a=np.asarray(Image.open(f'{D}/{split}/{p}').convert('RGB')).astype(np.float32); g=a.mean(2)
    r.append(np.abs(g[1:-1,1:-1]*4-g[:-2,1:-1]-g[2:,1:-1]-g[1:-1,:-2]-g[1:-1,2:]).mean())
  df['sharp']=r; return df
for s in ['train','public_test','private_test']: stats(s).to_pickle(f'{s}.pkl')
