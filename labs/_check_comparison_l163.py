"""Selection must be invariant to held-out targets; every requested test key appears."""
import copy,numpy as np
from _run_l163 import fit_heads,evaluate_heads
from relkit.rows_l163 import typed_features,choose_alpha
rows=[dict(id=str(i),price_usd=float(i),weight_kg=1.,colour='red',condition='new',target=2.*i) for i in range(12)]
p=dict(experiment='test',scope='fixture',rows=rows,splits=[dict(seed=0,train=list(range(6)),validation=[6,7,8],test=[9,10,11])],alphas=[.01,1,10],variants=['baseline','reordered','renamed'])
p['splits']=[dict(p['splits'][0],seed=s) for s in range(3)]
a={k:np.array([[i,i%2] for i in range(12)],dtype=float) for k in p['variants']}
h=fit_heads(p,a,typed_features,choose_alpha)
q=copy.deepcopy(p)
for i in q['splits'][0]['test']:q['rows'][i]['target']=-99999.
assert h==fit_heads(q,a,typed_features,choose_alpha),'Test targets affected model selection'
r=evaluate_heads(p,a,h,typed_features)
assert len(r['predictions'])==54
assert all(v['prediction_change']==0 for v in r['results'])
print('PASS: test-target intervention cannot change frozen heads; paired coverage complete')
