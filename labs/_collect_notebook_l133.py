"""Collect and independently verify the separate fresh portable GPU execution."""
import hashlib,json,math
from pathlib import Path
import modal
import numpy as np
import nbformat
P=Path(__file__).resolve().parent;volume=modal.Volume.from_name('l133-notebook-validation')
def fetch(remote,dest):
 dest.parent.mkdir(parents=True,exist_ok=True)
 tmp=dest.with_suffix(dest.suffix+'.partial')
 with tmp.open('wb') as f:
  for chunk in volume.read_file(remote):f.write(chunk)
 tmp.replace(dest)
root=P/'evidence/l133/notebook';fetch('report.json',root/'report.json');r=json.loads((root/'report.json').read_text());assert r['status']=='PASS' and r['full_gate']
code='\n\n'.join(c.source for c in nbformat.read(P/'solutions/0133-hetero-conv-reg.ipynb',as_version=4).cells if c.cell_type=='code');assert hashlib.sha256(code.encode()).hexdigest()==r['code_sha256']
fetch('work/l133-full/packet.json',root/'packet.json');fetch('work/l133-full/layer-traces.json',root/'layer-traces.json')
packet=json.loads((root/'packet.json').read_text());traces=json.loads((root/'layer-traces.json').read_text());assert len(traces)==5
assert [x['seed'] for x in packet['records']]==list(range(5))
count=0;maxerr=0
for seed in range(5):
 for name in ['result.json','predictions.npz','selected.pt']:
  dst=(P/'results/l133/notebook' if name=='selected.pt' else root)/f'seed-{seed}'/name
  fetch(f'work/l133-full/seed-{seed}/{name}',dst)
 result=json.loads((root/f'seed-{seed}/result.json').read_text());a=np.load(root/f'seed-{seed}/predictions.npz')
 assert result['epochs']==10 and len(result['trace'])==10 and all(x['train_queries']==7453 for x in result['trace'])
 assert result['selected_epoch']==min(result['trace'],key=lambda x:x['val_mae'])['epoch']
 assert result['checkpoint_sha256']==hashlib.sha256((P/f'results/l133/notebook/seed-{seed}/selected.pt').read_bytes()).hexdigest()
 for split,n in [('val',499),('test',760)]:
  assert len(a[split+'_pred'])==n
  score=math.fsum(abs(float(x)-float(y)) for x,y in zip(a[split+'_pred'],a[split+'_target']))/n
  assert abs(score-result['scores'][split])<1e-12 and abs(score-packet['records'][seed][split])<1e-12
  primary=np.load(P/f'evidence/l133/paper/seed-{seed}/predictions.npz')
  for suffix in ['entity','time','target']:np.testing.assert_array_equal(a[split+'_'+suffix],primary[split+'_'+suffix])
  assert result['replay'][split]['max_original_logit_error']<1e-5
  maxerr=max(maxerr,result['replay'][split]['max_original_logit_error']);count+=n
 t=traces[seed];assert t['status']=='PASS' and t['audit_dtype']=='float64' and t['audit_device']=='cpu'
 assert t['output_max_error']<1e-5 and t['gradient_max_error']<1e-5
b=json.loads((P/'_budget_l133.json').read_text());assert sum(x['workers'] for x in b['reservations'])==8
b['successful_worker_resource_usd']=b['recorded_worker_resource_usd']+r['resource_usd']
b['failed_pilot_worker_bound_usd']=.812592
b['resource_usd_including_failed_worker_bound']=b['successful_worker_resource_usd']+b['failed_pilot_worker_bound_usd']
assert b['resource_usd_including_failed_worker_bound']+b['overhead_reserve_usd']<10
b['status']='COMPLETE_NO_MORE_DISPATCH';b['notebook_worker_resource_usd']=r['resource_usd']
(P/'_budget_l133.json').write_text(json.dumps(b,indent=2)+'\n')
out=dict(status='PASS',code_sha256=r['code_sha256'],code_cells=r['code_cells'],full_gate='FIVE_FULL_FRESH_FITS',all_epochs='COMPLETE',predictions=count,trace_count=5,maximum_original_output_error=maxerr,worker_resource_usd=r['resource_usd'],prior_attempt='NONE for notebook; original author pilot failure retained separately',live_colab='NOT_CHECKED')
(P/'_notebook_gpu_l133_results.json').write_text(json.dumps(out,indent=2));print(out)
