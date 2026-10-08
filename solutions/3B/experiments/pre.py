import sys; sys.path.insert(0,'/tmp/claude-1000/-home-minh-Desktop-olympiad-ai/1f52811c-e600-457a-bc8e-50e6365e680a/scratchpad')
from feats import *; import pickle,time
D='/home/minh/Desktop/olympiad_ai/warmup/3B_molecule_binding/dataset/'
t=time.time(); res={}
for k,p in [('tr','train'),('pu','public_test'),('pr','private_test')]:
    res[k]=featurize_df(pd.read_csv(D+f'{p}/{p}.csv'))
print(time.time()-t)
pickle.dump(res,open('/tmp/claude-1000/-home-minh-Desktop-olympiad-ai/1f52811c-e600-457a-bc8e-50e6365e680a/scratchpad/cache.pkl','wb'),protocol=4)
