import re, unicodedata
import numpy as np, pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold
D='/home/minh/Desktop/olympiad_ai/warmup/2B_rice_multimodal/dataset'
CLASSES=["healthy","blast","brown_spot","bacterial_blight","nitrogen_deficiency"]
NUM=["days_after_sowing","humidity_7d","temp_max_7d","temp_min_7d","rainfall_7d_mm","storm_last_3d","nitrogen_kg_ha","field_area_ha"]
CAT=["soil_type","variety","season"]
def load():
    tr=pd.read_csv(f'{D}/train/train.csv'); pu=pd.read_csv(f'{D}/public_test/public_test.csv'); pr=pd.read_csv(f'{D}/private_test/private_test.csv')
    te=pd.concat([pu,pr],ignore_index=True); y=tr.label.map(CLASSES.index).values
    folds=list(StratifiedKFold(5,shuffle=True,random_state=0).split(tr,y))
    return tr,te,y,folds,len(pu)
def strip_accents(s):
    s=s.lower().replace('đ','d'); s=unicodedata.normalize('NFKD',s); s=''.join(c for c in s if not unicodedata.combining(c))
    s=re.sub(r'[^a-z0-9 ]+',' ',s); s=re.sub(r'(.)\1+',r'\1',s); return re.sub(r'\s+',' ',s).strip()
def make_cleaner(provinces):
    provs=sorted({strip_accents(p) for p in provinces},key=len,reverse=True)
    def clean(s):
        s=' '+strip_accents(s)+' '
        for p in provs: s=s.replace(' '+p+' ',' tinhx ')
        s=re.sub(r'\b\d+\b',' numx ',s)
        return re.sub(r'\s+',' ',s).strip()
    return clean
