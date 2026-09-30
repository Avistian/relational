"""Independent keyed metrics, complete-seed accounting and paired aggregation."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error
from relkit.comparison_l146 import select_config,paired_errors
P=Path(__file__).resolve().parent;E=P/'evidence/l146';runs=[];verified=0;differences={'val':[],'test':[]}
for seed in [0,1,2]:
 packets={}
 for arm in ['gnn','relgt']:
  root=E/f'fit-{arm}-{seed}';r=json.loads((root/'result.json').read_text());z=np.load(root/'predictions.npz')
  assert r['seed']==seed and r['arm']==arm and not r['pilot']
  assert len(r['history'])==10 and all(h['queries']==7453 for h in r['history'])
  assert r['selected_epoch']==select_config(list(range(1,11)),[h['val_mae'] for h in r['history']])
  assert not r['first_batch_nonfinite_gradients'] and r['repeat_eval_max_error']<=1e-5
  for split,n in [('val',499),('test',760)]:
   ref=np.load(E/f'prepared/{split}.npz');keys=list(zip(z[split+'_entity'],z[split+'_cutoff']))
   lookup=dict(zip(zip(ref['entity'],ref['cutoff']),ref['target']));assert len(keys)==len(set(keys))==n and set(keys)==set(lookup)
   np.testing.assert_array_equal([lookup[k] for k in keys],z[split+'_target'])
   metric=mean_absolute_error(z[split+'_target'],z[split+'_pred']);assert abs(metric-r['scores'][split])<1e-10;verified+=n
  runs.append(r);packets[arm]=z
 for split in ['val','test']:
  a,b=packets['gnn'],packets['relgt'];ak=list(zip(a[split+'_entity'],a[split+'_cutoff']));bk=list(zip(b[split+'_entity'],b[split+'_cutoff']))
  diff=paired_errors(ak,a[split+'_target'],ak,a[split+'_pred'],bk,b[split+'_pred']);differences[split].append(float(diff.mean()))
assert len({r['run_id'] for r in runs})==6
arms={}
for arm in ['gnn','relgt']:
 rr=[r for r in runs if r['arm']==arm];assert len({r['parameters'] for r in rr})==1
 arms[arm]=dict(parameters=rr[0]['parameters'],fit_seconds=sum(r['seconds'] for r in rr))
 for split in ['val','test']:
  vals=[r['scores'][split] for r in rr];arms[arm][split]=dict(mean=float(np.mean(vals)),sample_sd=float(np.std(vals,ddof=1)),values=vals)
paired={s:dict(mean=float(np.mean(d)),sample_sd=float(np.std(d,ddof=1)),differences=d,relgt_wins=sum(v>0 for v in d)) for s,d in differences.items()}
r=dict(status='PASS',course_experiment='COMPLETE',runs=runs,arms=arms,paired=paired,verified_predictions=verified,
       full_selected_reproduction='INCOMPLETE',fresh_canonical_rdl='NOT_RUN',full_relgt_search='NOT_RUN',historical_identity='NOT_ESTABLISHED',whole_paper='NOT_RUN')
(E/'summary.json').write_text(json.dumps(r,indent=2));print({k:r[k] for k in ['arms','paired','verified_predictions']})
