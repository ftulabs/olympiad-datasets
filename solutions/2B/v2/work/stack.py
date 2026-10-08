"""Stack experts' OOF log-probs. usage: python stack.py expertA expertB ...  (names: text, tab, v1_img_tiny, or remote tags)"""
import sys, glob, os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, log_loss
from common import *
HERE=os.path.dirname(os.path.abspath(__file__))
tr,te,y,folds,npu=load()
SC=os.path.join(HERE,'')  # sharpness pickles from sharpness_stats.py (run it in work/ first)
def sharp(split):
    return pd.read_pickle(SC+split+'.pkl').sharp.values
blur_tr=sharp('train')<2.5; blur_te=np.concatenate([sharp('public_test'),sharp('private_test')])<2.5
def load_expert(name):
    if name in ('text','tab'):
        z=np.load(f'{HERE}/oof/{name}.npz'); return z['oof'],z['test']
    if name.startswith('v1_'):
        V='/home/minh/Desktop/olympiad_ai/solutions/2B/oof/'; n=name[3:]
        return np.load(V+f'oof_{n}.npy'), np.load(V+f'test_{n}.npy')
    # remote tags may be '+'-joined (seed average)
    oofs,tests=[],[]
    for t in name.split('+'):
        fs=sorted(glob.glob(f'{HERE}/remote_out/{t}_f[0-4].npz')); assert len(fs)==5,(t,len(fs))
        oof=np.zeros((len(y),5)); tp=0
        for f in fs:
            z=np.load(f); oof[z['idx']]=z['oof']; tp=tp+z['test']/5
        oofs.append(oof); tests.append(tp)
    return np.mean(oofs,0), np.mean(tests,0)
L=lambda p: np.log(np.clip(p,1e-4,1))
def feats(ps, blur, gate):
    Z=np.hstack([L(p) for p in ps])
    if gate: Z=np.hstack([Z, Z*blur[:,None], blur[:,None]])
    return Z
def mf1(yy,p,w=None): return f1_score(yy,p.argmax(1),average='macro',sample_weight=w)
def evaluate(names, C=0.3, gate=False, verbose=True, seeds=(123,)):
    ex=[load_expert(n) for n in names]
    Ztr=feats([o for o,_ in ex],blur_tr,gate); Zte=feats([t for _,t in ex],blur_te,gate)
    cvs=[]
    for s in seeds:
        cv=np.zeros((len(y),5))
        for a,b in StratifiedKFold(5,shuffle=True,random_state=s).split(Ztr,y):
            cv[b]=LogisticRegression(C=C,max_iter=5000).fit(Ztr[a],y[a]).predict_proba(Ztr[b])
        cvs.append(cv)
    cv=np.mean(cvs,0)
    w=np.where(blur_tr,0.30/blur_tr.mean(),0.70/(1-blur_tr.mean()))
    res=dict(cv=mf1(y,cv), cv_w=mf1(y,cv,w), sharp=mf1(y[~blur_tr],cv[~blur_tr]), blur=mf1(y[blur_tr],cv[blur_tr]), ll=log_loss(y,cv))
    if verbose:
        print(f"{'+'.join(names)[:90]:90s} C={C} gate={int(gate)} | CV {res['cv']:.4f} wCV {res['cv_w']:.4f} sharp {res['sharp']:.4f} blur {res['blur']:.4f} ll {res['ll']:.4f}",flush=True)
    pte=LogisticRegression(C=C,max_iter=5000).fit(Ztr,y).predict_proba(Zte)
    return res, cv, pte
if __name__=='__main__':
    for n in sys.argv[1:]:
        o,_=load_expert(n); print(n,'F1 %.4f sharp %.4f blur %.4f ll %.4f'%(mf1(y,o),mf1(y[~blur_tr],o[~blur_tr]),mf1(y[blur_tr],o[blur_tr]),log_loss(y,np.clip(o,1e-6,1))))
    evaluate(sys.argv[1:]); evaluate(sys.argv[1:],gate=True)
