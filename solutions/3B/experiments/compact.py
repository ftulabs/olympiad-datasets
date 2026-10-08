import pickle, numpy as np
W='/tmp/claude-1000/-home-minh-Desktop-olympiad-ai/1f52811c-e600-457a-bc8e-50e6365e680a/scratchpad/w3B'
res=pickle.load(open(f'{W}/cache.pkl','rb'))
out={}
for k in ['tr','pu','pr']:
    L=res[k]
    na=np.array([len(m['part']) for m in L]); ne=np.array([len(m['e']) for m in L])
    out[k+'_na']=na; out[k+'_ne']=ne
    out[k+'_x']=np.concatenate([m['x'] for m in L]).astype(np.uint8)
    out[k+'_part']=np.concatenate([m['part'] for m in L]).astype(np.int8)
    out[k+'_e']=np.concatenate([m['e'] for m in L]).astype(np.int16)
    out[k+'_keys']=np.concatenate([m['keys'] for m in L],1)
    out[k+'_scaf']=np.array([m['scaf'] for m in L],np.uint64)
    out[k+'_blocks']=np.array([m['blocks'] for m in L],np.uint64)
np.savez(f'{W}/cache.npz',**out)
