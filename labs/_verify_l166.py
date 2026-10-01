"""Independent mathematical oracles and deliberately incorrect student solutions."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import itertools,json,tempfile,hashlib
from pathlib import Path
import numpy as np
import torch
from relkit.rdbpfn_l166 import relational_prior,dfs_summary,context_mask,normalize_support
from _check_l166 import check166
from _guard_l166 import reserve
torch.set_num_threads(1)
assert check166(relational_prior,dfs_summary,context_mask)=='PASS'
aggregations=0;masks=0
for fk in itertools.product(range(3),repeat=4):
    values=np.array([2.,-1.,3.,9.]);expected=[]
    for parent in [0,1,2,3]:
        selected=[v for key,v in zip(fk,values) if key==parent]
        expected.append([len(selected),sum(selected)/len(selected) if selected else 0])
    np.testing.assert_allclose(dfs_summary([0,1,2,3],fk,values),expected);aggregations+=1
for rows in range(1,10):
    for support in range(1,rows+1):
        m=context_mask(rows,support)
        for i,j in itertools.product(range(rows),repeat=2):assert bool(m[i,j])==(j in range(support))
        masks+=1
x=torch.tensor([[[1.,float('nan')],[3.,float('nan')],[1000.,9.]]],dtype=torch.float64)
expected=np.array([[[[-1.],[0.]],[[1.],[0.]],[[100.],[100.]]]])
np.testing.assert_allclose(normalize_support(x,2).numpy(),expected)
wrong=[(relational_prior,lambda a,b,c:np.zeros((len(a),2)),context_mask),
       (relational_prior,dfs_summary,lambda n,s:np.ones((n,n),bool)),
       (lambda seed,parents,children,strength:relational_prior(seed,parents,children,0.),dfs_summary,context_mask)]
for functions in wrong:
    try:check166(*functions)
    except (AssertionError,ValueError):pass
    else:raise AssertionError('Incorrect learner function passed')
P=Path(__file__).resolve().parent;root=P.parent
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)/'budget.json';p.write_text(json.dumps(dict(cap_usd=10,overhead_reserve_usd=2,reservations=[],source_hashes={})))
    reserve(p,root,'one',600)
    for phase,secs in [('one',600),('huge',40000),('invalid',-1)]:
        try:reserve(p,root,phase,secs)
        except ValueError:pass
        else:raise AssertionError('Invalid dispatch accepted')
    b=json.loads(p.read_text());b['source_hashes']={'labs/_run_l166.py':'wrong'};p.write_text(json.dumps(b))
    try:reserve(p,root,'changed',600)
    except ValueError:pass
    else:raise AssertionError('Changed source accepted')
r=dict(status='PASS',independent_group_aggregation_cases=aggregations,independent_mask_cases=masks,support_normalization='PASS',incorrect_learner_functions_rejected=3,budget_guards=4)
(P/'_verify_l166_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
