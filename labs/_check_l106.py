"""Semantic checks: first written before implementation; independent metric oracle."""
import importlib.util,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('eb',P/'relkit/edgebank_l106.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
np.testing.assert_array_equal(m.legal_history(np.array([1,2,3,3]),np.array([1,4,3,3]),3),[True,False,False,False])
h=np.array([[1,10,1],[2,11,2],[1,10,3],[3,12,10]],float)
c=np.array([[1,10],[10,1],[2,11],[3,12],[4,13]])
np.testing.assert_array_equal(m.memory_scores(h,c),[1,0,1,1,0])
np.testing.assert_array_equal(m.memory_scores(h,c,'window'),[0,0,0,1,0])
np.testing.assert_array_equal(m.memory_scores(np.empty((0,3)),c),np.zeros(5))
assert np.allclose(m.binary_metrics([1,0],[0,0]),[.75,.75])
assert np.allclose(m.binary_metrics([1,0],[1,1]),[5/12,.25])
rng=np.random.default_rng(106)
for _ in range(100):
 p=rng.integers(0,2,rng.integers(1,30));n=rng.integers(0,2,rng.integers(1,30));y=np.r_[np.ones(len(p)),np.zeros(len(n))];s=np.r_[p,n]
 assert np.allclose(m.binary_metrics(p,n),[average_precision_score(y,s),roc_auc_score(y,s)])
for fn,args in [(m.memory_scores,(h,c,'bad')),(m.binary_metrics,([], [0])),(m.legal_history,([1],[0],2))]:
 try:fn(*args)
 except ValueError:pass
 else:raise AssertionError('Invalid inputs must fail')
print('PASS: causal boundary, directed identity, recency, empty memory, 100 metric oracles, invalid inputs')
(P/'_check_l106_results.json').write_text(json.dumps({'status':'PASS','random_metric_oracles':100,'causal_ties_and_late_arrival':'PASS','memory_cases':'PASS'},indent=2)+'\n')
