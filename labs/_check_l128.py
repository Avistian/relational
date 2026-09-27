"""Behavioral checks independent of implementation; also used in portable notebooks."""
import math,json
from pathlib import Path
from relkit.taxonomy_l128 import task_contract,binary_auc,ranking_map

def rejected(fn,*args):
    try:fn(*args)
    except ValueError:return
    raise AssertionError('Invalid input was accepted')

def check_contract(fn):
    a=fn('binary');b=fn('regression');c=fn('recommendation')
    assert (a['head'],a['loss'],a['metric'],a['maximize'])==('one_logit','BCEWithLogits','roc_auc',True)
    assert (b['head'],b['loss'],b['metric'],b['maximize'])==('one_scalar','L1','mae',False)
    assert (c['head'],c['loss'],c['metric'],c['maximize'])==('candidate_scores','BPR','map',True)
    assert all(x['split']=='temporal' for x in [a,b,c])
    rejected(fn,'autoregressive');rejected(fn,'temporal')
    a['metric']='changed';assert fn('binary')['metric']=='roc_auc'

def check_auc(fn):
    assert fn([0,1,0,1],[.1,.4,.4,.8])==.875
    assert fn([0,1],[1,1])==.5
    assert fn([0,1],[1,0])==0
    assert fn([0,1],[0,1])==1
    rejected(fn,[1,1],[.1,.2]);rejected(fn,[0,2],[0,1]);rejected(fn,[0,1],[1]);rejected(fn,[0,1],[0,float('nan')])
    from sklearn.metrics import roc_auc_score
    import numpy as np
    rng=np.random.default_rng(128)
    for _ in range(80):
        y=np.r_[0,1,rng.integers(0,2,20)];s=rng.integers(0,5,len(y))/4
        assert abs(fn(y,s)-roc_auc_score(y,s))<1e-12

def check_map(fn):
    # q0 AP=(1 + 2/3)/2; q1 AP=1/2; q2 excluded (zero positives).
    assert abs(fn([{2,4},{9},set()],[[2,7,4],[8,9,3],[1,2,3]],3)-2/3)<1e-12
    assert fn([{1,2,3,4}],[[1,2]],2)==1 # denominator min(k, number relevant)
    rejected(fn,[{1}],[[1,1]],2);rejected(fn,[{1}],[[1]],2)
    rejected(fn,[set()],[[1]],1);rejected(fn,[{1}],[[1]],0)
    from relbench.metrics import link_prediction_map
    import numpy as np
    rng=np.random.default_rng(128)
    for _ in range(40):
        truth=[set(rng.choice(20,size=int(rng.integers(0,10)),replace=False).tolist()) for _ in range(8)]
        ranked=[rng.choice(20,size=5,replace=False).tolist() for _ in truth]
        hit=np.array([[v in t for v in row] for t,row in zip(truth,ranked)])
        count=np.array([len(t) for t in truth]);assert abs(fn(truth,ranked,5)-link_prediction_map(hit,count))<1e-12

if __name__=='__main__':
    for check,fn in [(check_contract,task_contract),(check_auc,binary_auc),(check_map,ranking_map)]:check(fn)
    r=dict(status='PASS',auc_comparisons=80,map_comparisons=40,invalid_inputs='REJECTED')
    Path(__file__).with_name('_check_l128_results.json').write_text(json.dumps(r,indent=2));print(r)
