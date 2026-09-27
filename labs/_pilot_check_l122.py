"""Gate full dispatch on complete, source-checked pilot and conservative cost."""
import hashlib,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
root=P/'evidence/l122/pilot/seed-100'
r=json.loads((root/'result.json').read_text());a=json.loads((root/'audit.json').read_text());d=json.loads((root/'completed.json').read_text())
assert d['lesson']==122 and d['seed']==100 and d['status']=='COMPLETE'
assert r['epochs']==1 and r['trace'][0]['train_queries']==7453
assert sum(a['rows'].values())==74063
assert r['checkpoint_sha256']==hashlib.sha256((P/'results/l122/pilot/seed-100/selected.pt').read_bytes()).hexdigest()
x=np.load(root/'predictions.npz')
for split,n in [('val',499),('test',760)]:
 assert len(x[split+'_pred'])==n
 assert abs(math.fsum(abs(float(p)-float(y)) for p,y in zip(x[split+'_pred'],x[split+'_target']))/n-r['scores'][split])<1e-12
 assert r['replay'][split]['max_original_logit_error']<1e-5
b=json.loads((P/'_budget_l122.json').read_text())
for relative,digest in b['source_hashes'].items():assert hashlib.sha256((P.parent/relative).read_bytes()).hexdigest()==digest
# Charge ten times the entire pilot (including setup/final eval) per full fit, conservatively.
projection=d['resource_usd']*(1+5*10)
assert projection< b['maximum_worker_usd'] and d['seconds']*10<3600
b['pilot_approved_for_full']=True;b['pilot_projection_worker_usd']=projection
(P/'_budget_l122.json').write_text(json.dumps(b,indent=2)+'\n')
out={'status':'PASS','pilot_worker_usd':d['resource_usd'],'conservative_pilot_plus_five_fit_worker_projection':projection,'reservation_ceiling_usd':b['maximum_worker_usd'],'overhead_reserve_usd':b['overhead_reserve_usd'],'gate':'Full data, hashes, checkpoint, all final queries and original model replay passed'}
(P/'_pilot_l122_results.json').write_text(json.dumps(out,indent=2));print(out)
