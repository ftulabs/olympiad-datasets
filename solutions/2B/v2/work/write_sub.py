import sys, numpy as np, pandas as pd, stack as S
out, names = sys.argv[1], sys.argv[2:]
r, cv, pte = S.evaluate(names, C=0.3, seeds=(1, 2, 3))
lab = np.array(S.CLASSES)[pte.argmax(1)]; n = S.npu
pd.DataFrame({'id': S.te.id[:n], 'label': lab[:n]}).to_csv(f'{out}_pub.csv', index=False)
pd.DataFrame({'id': S.te.id[n:], 'label': lab[n:]}).to_csv(f'{out}_priv.csv', index=False)
np.save(f'{out}_testprob.npy', pte)
print('mix', pd.Series(lab).value_counts(normalize=True).round(3).to_dict())
