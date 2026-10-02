"""Compare identical 1024-example supports across old and new inference; no score selection."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;E=P/'evidence/l176';rows=[]
for db in ['rel-f1','rel-trial']:
 for arm in ['RDBPFN','RDBPFN_single','TabICLv1.1']:
  for seed in range(10):
   name=f'{db}-{arm}-1024-{seed}.npz'
   old=P/'evidence/l169'/('pilot-1' if seed==0 else 'remaining-complete')/name
   new=E/('pilot-1' if seed==0 else 'remaining-1')/name
   a=np.load(old);b=np.load(new)
   for key in ['keys','label','support_keys']:np.testing.assert_array_equal(a[key],b[key])
   delta=float(np.max(np.abs(a['probability']-b['probability'])))
   pos=b['label']==1;neg=~pos
   def auc(p):return float(np.mean((p[pos,None]>p[None,neg])+.5*(p[pos,None]==p[None,neg])))
   rows.append(dict(database=db,arm=arm,seed=seed,max_probability_difference=delta,auc_difference=auc(b['probability'])-auc(a['probability']),exact=delta==0))
r=dict(status='COMPLETE_SHARED_SUPPORT_COMPARISON',evaluations=60,exact=sum(x['exact'] for x in rows),max_probability_difference=max(x['max_probability_difference'] for x in rows),max_absolute_auc_difference=max(abs(x['auc_difference']) for x in rows),rows=rows,interpretation='Descriptive consistency at identical1024supports; no test-score selection or claim of exact TabICL repeatability')
(E/'shared-1024-comparison.json').write_text(json.dumps(r,indent=2)+'\n')
print({k:v for k,v in r.items() if k!='rows'})
for name,c in json.loads((E/'report.json').read_text())['curves'].items():
 print(name,[round(x['mean'],6) for x in c['levels']]);print('gains',[(x['start'],x['end'],round(x['mean_gain'],6),x['positive_seeds']) for x in c['doublings']])
