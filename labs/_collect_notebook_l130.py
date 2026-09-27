"""Collect and independently score the successful full portable training gate."""
import json,hashlib,math
from pathlib import Path
import modal,numpy as np
from relkit.checkpoint_l130 import reproduction_verdict
P=Path(__file__).resolve().parent;out=P/'evidence/l130/notebook';out.mkdir(parents=True,exist_ok=True)
v=modal.Volume.from_name('l130-notebook-validation')
paths=['report.json','work/l130-full/packet.json']
paths += [f'work/l130-full/seed-{s}/{name}' for s in range(5) for name in ['result.json','predictions.npz']]
for remote in paths:
 local=out/remote.replace('work/l130-full/','');local.parent.mkdir(parents=True,exist_ok=True)
 tmp=local.with_suffix(local.suffix+'.partial')
 with tmp.open('wb') as f:
  for chunk in v.read_file(remote):f.write(chunk)
 tmp.replace(local)
report=json.loads((out/'report.json').read_text());assert report['status']=='PASS'
nb=json.loads((P/'solutions/0130-rdl-checkpoint.ipynb').read_text());code='\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
assert hashlib.sha256(code.encode()).hexdigest()==report['code_sha256']
packet=json.loads((out/'packet.json').read_text());count=0
for row in packet['records']:
 seed=row['seed'];r=json.loads((out/f'seed-{seed}/result.json').read_text());a=np.load(out/f'seed-{seed}/predictions.npz');original=np.load(P/f'evidence/l130/paper/seed-{seed}/predictions.npz')
 assert r['epochs']==10 and len(r['trace'])==10 and all(x['train_queries']==7453 for x in r['trace'])
 assert min(r['trace'],key=lambda x:x['val_mae'])['epoch']==r['selected_epoch']
 for split,n in [('val',499),('test',760)]:
  for suffix in ['entity','time','target']:np.testing.assert_array_equal(a[split+'_'+suffix],original[split+'_'+suffix])
  score=math.fsum(abs(float(p)-float(y)) for p,y in zip(a[split+'_pred'],a[split+'_target']))/n
  assert abs(score-r['scores'][split])<1e-12 and abs(score-row[split])<1e-12
  assert r['replay'][split]['max_original_logit_error']<1e-5;count+=n
assert reproduction_verdict(packet['records'])==packet['metrics']==report['packet']['metrics']
bp=P/'_budget_l130.json';b=json.loads(bp.read_text());b['notebook_validation_worker_usd']=report['resource_usd']
b['successful_worker_resource_usd']=b['recorded_worker_resource_usd']+report['resource_usd']
b['resource_usd_including_failed_worker_bound']=b['successful_worker_resource_usd']+b['validation_attempt1']['worker_cost_upper_bound_usd']
assert b['resource_usd_including_failed_worker_bound']+b['overhead_reserve_usd']<10
assert sum(x['workers'] for x in b['reservations'])==8
b['status']='COMPLETE_NO_MORE_DISPATCH';bp.write_text(json.dumps(b,indent=2)+'\n')
r=dict(status='PASS',all_code_cells=report['code_cells'],code_sha256=report['code_sha256'],full_gate='FIVE_FULL_FRESH_FITS',independently_scored_predictions=count,metrics=packet['metrics'],successful_worker_resource_usd=b['successful_worker_resource_usd'],failed_validation_worker_upper_bound=b['validation_attempt1']['worker_cost_upper_bound_usd'],primary_experiment='Separate five-seed evidence; gate validation never enters its aggregate',live_colab='NOT_CHECKED')
(P/'_notebook_gate_l130_results.json').write_text(json.dumps(r,indent=2));print(r)
