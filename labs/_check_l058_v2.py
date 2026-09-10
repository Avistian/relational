"""Diagnostic function contracts used verbatim in the student lab."""
CHECKS={
'parse_talent':'''# CHECK — real cell plus malformed and missing-data fixtures
means,sds=parse_talent(ROOT/'sources/l058/cls_bin.md')
assert means.shape==(120,31) and means.index.is_unique
assert means.loc['Pima_Indians_Diabetes_Database','xgboost']==.7645
assert sds.loc['Pima_Indians_Diabetes_Database','xgboost']==.0110
from tempfile import TemporaryDirectory
with TemporaryDirectory() as temp:
    path=Path(temp)/'scores.md'
    head='| Dataset | A | B |\\n| --- | --- | --- |\\n'
    path.write_text(head+'| d | **-1.2e-3+2e-4** | nan+nan |\\n')
    m,s=parse_talent(path);assert m.loc['d','A']==-.0012 and s.loc['d','A']==.0002 and np.isnan(m.loc['d','B'])
    for body in ['| d | typo | .1+.2 |\\n','| d | .1+-.2 | .1+.2 |\\n','| d | .1+.2 | .1+.2 |\\n'*2]:
        path.write_text(head+body)
        try:parse_talent(path)
        except ValueError:pass
        else:raise AssertionError('Reject malformed values, negative SDs, duplicate IDs; preserve only declared missing markers')
print('PASS: actual source, mean/SD, missingness, malformed cells and identities')''',
'rank_matrix':'''# CHECK — metric direction, rounded ties, scale and pool
x=np.array([[.1,.2,.2,.9],[3,1,2,4]])
np.testing.assert_array_equal(rank_matrix(x,False),[[1,2.5,2.5,4],[3,1,2,4]])
np.testing.assert_allclose(rank_matrix(x,True),5-rank_matrix(x,False))
np.testing.assert_array_equal(rank_matrix(x*100,False),rank_matrix(x,False))
try:rank_matrix([[1,np.nan]],False)
except ValueError:pass
else:raise AssertionError('Missing scores cannot silently enter a complete rank panel')
print('PASS: orient first, rank within rows, keep average ties')''',
'dataset_bootstrap':'''# CHECK — shared dataset indices and reproducible percentiles
x=np.array([[1.,3.],[2.,2.],[3.,1.]])
ci=dataset_bootstrap(x,2000,58)
np.testing.assert_allclose(ci[:,0],4-ci[::-1,1],atol=1e-12)
np.testing.assert_array_equal(ci,dataset_bootstrap(x,2000,58))
np.testing.assert_array_equal(dataset_bootstrap(np.ones((8,2)),40,58),np.ones((2,2)))
for bad in [np.array([]),np.array([[1,np.nan]])]:
    try:dataset_bootstrap(bad)
    except ValueError:pass
    else:raise AssertionError('Require a finite, nonempty 2D panel')
print('PASS: paired rows, constant-case interval, deterministic draws')''',
'select_tiny':'''# CHECK — without-replacement proposals, MAE and nested budget
x=np.array([[1,2],[1,2],[2,1],[2,1],[1,2],[2,1]],float)
a=select_tiny(x,2,1,58);b=select_tiny(x,2,100,58)
assert len(set(b['indices']))==2 and min(b['indices'])>=0 and max(b['indices'])<6
assert b['seen_mae']<=a['seen_mae'] and b['first_indices']==a['first_indices']
assert all(a>=b for a,b in zip(b['best_so_far'],b['best_so_far'][1:]))
assert b['seen_mae']==0, 'This fixed proposal sequence reaches a perfectly balanced subset'
z=select_tiny(np.ones((5,2)),2,100,58)
assert z['indices']==z['first_indices'], 'Retain the first equal minimum'
assert select_tiny(x,6,1,58)['seen_mae']==0
print('PASS: subset cardinality, first-minimum ties and nested search objective')''',
'method_holdout':'''# CHECK — changing held-out scores cannot select different task IDs
x=np.array([[1,2,1,2],[1,2,2,1],[2,1,1,2],[2,1,2,1],[1,2,1,2],[2,1,2,1]],float)
a=method_holdout(x,False,[0,1],2,100,58)
y=x.copy();y[:,2:]=np.arange(12).reshape(6,2)*17-30
b=method_holdout(y,False,[0,1],2,100,58)
assert a['selection']==b['selection'], 'Do not rank against hidden columns before selection'
assert a['seen']==[0,1] and a['unseen']==[2,3]
# Prove every public student function is called by the real driver later;
# this CHECK tests the most subtle indirect-leakage boundary specifically.
print('PASS: held-out methods cannot enter the selection computation')'''}
if __name__=='__main__':
 from pathlib import Path
 import numpy as np
 from relkit.talent_audit_l058 import *
 ROOT=Path(__file__).resolve().parent
 for source in CHECKS.values():exec(source)
 print('All five L058 behavioral contracts passed')
