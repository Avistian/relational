"""Collect and independently verify the separate fresh portable GPU execution."""
import hashlib,json,math
from pathlib import Path
import modal
import numpy as np
import nbformat
P=Path(__file__).resolve().parent;volume=modal.Volume.from_name('l134-notebook-validation')
def fetch(remote,dest):
 dest.parent.mkdir(parents=True,exist_ok=True)
 tmp=dest.with_suffix(dest.suffix+'.partial')
 with tmp.open('wb') as f:
  for chunk in volume.read_file(remote):f.write(chunk)
 tmp.replace(dest)
root=P/'evidence/l134/notebook';fetch('report.json',root/'report.json');r=json.loads((root/'report.json').read_text());assert r['status']=='PASS' and r['full_gate']
code='\n\n'.join(c.source for c in nbformat.read(P/'solutions/0134-training-at-scale.ipynb',as_version=4).cells if c.cell_type=='code');assert hashlib.sha256(code.encode()).hexdigest()==r['code_sha256']
fetch('work/l134-full/packet.json',root/'packet.json')
packet=json.loads((root/'packet.json').read_text())
assert [x['seed'] for x in packet['records']]==list(range(5))
count=0;maxerr=0
for seed in range(5):
 for name in ['result.json','predictions.npz','selected.pt']:
  dst=(P/'results/l134/notebook' if name=='selected.pt' else root)/f'seed-{seed}'/name
  fetch(f'work/l134-full/seed-{seed}/{name}',dst)
 result=json.loads((root/f'seed-{seed}/result.json').read_text());a=np.load(root/f'seed-{seed}/predictions.npz')
 assert result['epochs']==10 and len(result['trace'])==10 and all(x['train_queries']==7453 for x in result['trace'])
 assert result['selected_epoch']==min(result['trace'],key=lambda x:x['val_mae'])['epoch']
 assert result['checkpoint_sha256']==hashlib.sha256((P/f'results/l134/notebook/seed-{seed}/selected.pt').read_bytes()).hexdigest()
 for split,n in [('val',499),('test',760)]:
  assert len(a[split+'_pred'])==n
  score=math.fsum(abs(float(x)-float(y)) for x,y in zip(a[split+'_pred'],a[split+'_target']))/n
  assert abs(score-result['scores'][split])<1e-12 and abs(score-packet['records'][seed][split])<1e-12
  primary=np.load(P/f'evidence/l134/paper/seed-{seed}/predictions.npz')
  for suffix in ['entity','time','target']:np.testing.assert_array_equal(a[split+'_'+suffix],primary[split+'_'+suffix])
  assert result['replay'][split]['max_original_logit_error']<1e-5
  maxerr=max(maxerr,result['replay'][split]['max_original_logit_error']);count+=n
b=json.loads((P/'_budget_l134.json').read_text());assert sum(x['workers'] for x in b['reservations'])<=8
b['notebook_worker_resource_usd']=r['resource_usd']
b['successful_f1_worker_resource_usd']=b['recorded_successful_worker_resource_usd']+r['resource_usd']
b['scale_worker_resource_usd']=sum(json.loads(p.read_text())['resource_usd'] for p in (P/'evidence/l134/scale').glob('attempt-*/cost.json'))
b['reserved_scale_ceiling_usd']=sum(x['upper_bound_usd'] for x in b['scale_reservations'])
b['all_successful_and_failed_worker_estimate_usd']=b['successful_f1_worker_resource_usd']+b['scale_worker_resource_usd']
b['conservative_allocation_ceiling_usd']=b['maximum_worker_usd']+5+b['overhead_reserve_usd']
assert b['conservative_allocation_ceiling_usd']<=10
b['status']='COMPLETE_NO_MORE_DISPATCH'
(P/'_budget_l134.json').write_text(json.dumps(b,indent=2)+'\n')
out=dict(status='PASS',code_sha256=r['code_sha256'],code_cells=r['code_cells'],full_gate='FIVE_FULL_FRESH_FITS',all_epochs='COMPLETE',predictions=count,maximum_original_output_error=maxerr,worker_resource_usd=r['resource_usd'],prior_attempt='NONE for notebook; scale failures retained separately',live_colab='NOT_CHECKED')
(P/'_notebook_gpu_l134_results.json').write_text(json.dumps(out,indent=2));print(out)
