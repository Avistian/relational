"""Independent, diagnostic contract checks for the five live L060 operations."""
import numpy as np
CHECKS={
'validate_partitions':'''# CHECK — malformed structure and overlap cannot pass
assert validate_partitions({'train':[0,1],'val':[2],'test':[3]}) is True
for bad in [{'train':[0,0],'val':[2],'test':[3]}, {'train':[0],'val':[0],'test':[3]}, {'train':[[0]],'val':[2],'test':[3]}, {'train':[],'val':[2],'test':[3]}]:
    try: validate_partitions(bad)
    except ValueError: pass
    else: raise AssertionError('Reject duplicate, overlapping, non-vector or empty IDs')''',
'choose_validation':'''# CHECK — stable ties, no invalid candidate
assert choose_validation([.4,.2,.2]) == 1
for bad in [[],[.1,float('nan')],[[.2,.1]]]:
    try: choose_validation(bad)
    except ValueError: pass
    else: raise AssertionError('Reject empty, nonfinite or non-vector validation errors')''',
'score_predictions':'''# CHECK — explicit probability convention and target units
np.testing.assert_allclose(score_predictions([1,0],[.8,.3],False),(-np.log(.8)-np.log(.7))/2)
np.testing.assert_allclose(score_predictions([1,5],[4,1],True),np.sqrt(12.5))
for y,p,reg in [([0,1],[.2],False),([0,2],[.2,.8],False),([0,1],[.2,1.1],False),([0,1],[[.2],[.8]],False),([],[],True)]:
    try: score_predictions(y,p,reg)
    except ValueError: pass
    else: raise AssertionError('Reject misalignment, invalid labels/probabilities and empty data')''',
'aggregate_panel':'''# CHECK — mean seeds THEN rank; explicit roster detects wholly missing tasks
fixture=[dict(dataset=d,arm=a,seed=s,error=e,seconds=1.) for d,rows in [('d1',[[0.,1.],[.4,.4],[.7,.7]]),('d2',[[.2,.2],[.3,.3],[.4,.4]]),('d3',[[.9,.9],[.1,.1],[.5,.5]])] for a,errors in zip(['A','B','C'],rows) for s,e in enumerate(errors)]
a=aggregate_panel(fixture,['d1','d2','d3'],['A','B','C'],[0,1])
np.testing.assert_allclose(a['mean_ranks']['A'],2.)
np.testing.assert_allclose(a['mean_ranks']['B'],4/3)
for bad in [fixture[:-1],fixture+[fixture[0]], [r for r in fixture if r['dataset']!='d3']]:
    try: aggregate_panel(bad,['d1','d2','d3'],['A','B','C'],[0,1])
    except ValueError: pass
    else: raise AssertionError('Reject missing cells, duplicates and entirely absent declared datasets')''',
'paired_effect':'''# CHECK — pair by seed, not incoming row order; subtract before computing spread
fixture=[dict(dataset='d',arm=a,seed=s,error=e) for a,values in [('A',[.1,.4,.7]),('B',[.2,.3,.6])] for s,e in enumerate(values)]
r=paired_effect(list(reversed(fixture)),'d','A','B')
np.testing.assert_allclose(r['mean'],1/30)
np.testing.assert_allclose(r['sd'],np.std([-.1,.1,.1],ddof=1))
assert r['n']==3 and r['t95'][0] < r['mean'] < r['t95'][1]
try: paired_effect(fixture[:-1],'d','A','B')
except ValueError: pass
else: raise AssertionError('Do not pair unequal seed sets or truncate them')'''}
def run():
 from relkit import comparison_l060 as m
 scope=dict(vars(m))
 for name,check in CHECKS.items():exec(check,scope);print(name,'PASS')
if __name__=='__main__':
 import json,hashlib,copy
 from pathlib import Path
 from scipy.stats import rankdata
 from sklearn.metrics import log_loss,root_mean_squared_error
 from relkit.comparison_l060 import score_predictions,audit_records
 from relkit.tabm_v2 import TabM
 import torch
 run();root=Path(__file__).resolve().parent
 result=json.loads((root/'_verify_l060_v2_results.json').read_text())
 summary=audit_records(result);maximum=0.
 for r in result['records']:
    expected=root_mean_squared_error(r['targets'],r['predictions']) if r['metric']=='RMSE' else log_loss(r['targets'],r['predictions'],labels=[0,1])
    maximum=max(maximum,abs(expected-r['error']))
    assert r['class_order']==(None if r['metric']=='RMSE' else [0,1])
 assert maximum<1e-12
 for name,sha in result['source_hashes'].items():assert hashlib.sha256((root/'relkit'/name).read_bytes()).hexdigest()==sha, name
 for mutate in ['target','ids','selection','whole_dataset']:
    bad=copy.deepcopy(result)
    if mutate=='target':bad['records'][0]['targets'][0]=1-bad['records'][0]['targets'][0]
    elif mutate=='ids':bad['records'][0]['test_ids'].reverse()
    elif mutate=='selection':bad['records'][0]['selected']=1-bad['records'][0]['selected']
    else:bad['records']=[r for r in bad['records'] if r['dataset']!='diabetes/random']
    try:audit_records(bad)
    except ValueError:pass
    else:raise AssertionError('Mutation accepted: '+mutate)
 model=TabM(5,k=8,width=64,depth=3,arch='mini')
 assert all(b.S is None and b.bias.ndim==1 for b in model.blocks)
 assert model.blocks[0].R is not None and all(b.R is None for b in model.blocks[1:])
 assert all(b.weight.abs().max()<=1/(b.weight.shape[0]**.5) for b in model.blocks)
 prior=json.loads((root/'_source_check_l054_results.json').read_text())
 assert prior['implementation_sha256']==hashlib.sha256((root/'relkit/tabm_v2.py').read_bytes()).hexdigest()
 report={'status':'PASS','records':len(result['records']),'metric_library_max_error':maximum,'checks':list(CHECKS)+['four artifact mutations','all measured dependency hashes','correct mini topology/fan-in','L054 copied-weight parity implementation identity'],'paper_reproduction':'INCOMPARABLE'}
 (root/'_check_l060_v2_results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
