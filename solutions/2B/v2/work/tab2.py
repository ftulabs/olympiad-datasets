import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, log_loss
from common import *
from experts_tt import tab_matrix, CM
tr,te,y,folds,npu=load(); X=tab_matrix(tr,tr)
for leaf,it,lr in [(2,1500,0.02),(3,1200,0.01),(4,600,0.01)]:
    oof=np.zeros((len(y),5))
    for a,b in folds: oof[b]=HistGradientBoostingClassifier(learning_rate=lr,max_iter=it,max_leaf_nodes=leaf,min_samples_leaf=30,l2_regularization=1.0,categorical_features=CM,random_state=0).fit(X[a],y[a]).predict_proba(X[b])
    print(leaf,it,lr,'F1 %.4f ll %.4f'%(f1_score(y,oof.argmax(1),average='macro'),log_loss(y,oof)),flush=True)
