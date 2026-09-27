"""Collect and independently verify the separate fresh portable GPU execution."""
import hashlib,json,math
from pathlib import Path
import modal
import numpy as np
import nbformat
P=Path(__file__).resolve().parent;volume=modal.Volume.from_name('l131-notebook-recovery')
def fetch(remote,dest):
 dest.parent.mkdir(parents=True,exist_ok=True)
 tmp=dest.with_suffix(dest.suffix+'.partial')
 with tmp.open('wb') as f:
  for chunk in volume.read_file(remote):f.write(chunk)
 tmp.replace(dest)
root=P/'evidence/l131/notebook';fetch('report.json',root/'report.json');r=json.loads((root/'report.json').read_text());assert r['status']=='PASS' and r['full_gate']
code='\n\n'.join(c.source for c in nbformat.read(P/'solutions/0131-gnn-tabular-stack.ipynb',as_version=4).cells if c.cell_type=='code');assert hashlib.sha256(code.encode()).hexdigest()==r['code_sha256']
fetch('work/l131-full/packet.json',root/'packet.json');fetch('work/l131-full/stack-traces.json',root/'stack-traces.json')
packet=json.loads((root/'packet.json').read_text());traces=json.loads((root/'stack-traces.json').read_text());assert len(traces)==5
assert [x['seed'] for x in packet['records']]==list(range(5))
count=0;maxerr=0
for seed in range(5):
 for name in ['result.json','predictions.npz','selected.pt']:
  dst=(P/'results/l131/notebook' if name=='selected.pt' else root)/f'seed-{seed}'/name
  fetch(f'work/l131-full/seed-{seed}/{name}',dst)
 result=json.loads((root/f'seed-{seed}/result.json').read_text());a=np.load(root/f'seed-{seed}/predictions.npz')
 assert result['epochs']==10 and len(result['trace'])==10 and all(x['train_queries']==7453 for x in result['trace'])
 assert result['selected_epoch']==min(result['trace'],key=lambda x:x['val_mae'])['epoch']
 assert result['checkpoint_sha256']==hashlib.sha256((P/f'results/l131/notebook/seed-{seed}/selected.pt').read_bytes()).hexdigest()
 for split,n in [('val',499),('test',760)]:
  assert len(a[split+'_pred'])==n
  score=math.fsum(abs(float(x)-float(y)) for x,y in zip(a[split+'_pred'],a[split+'_target']))/n
  assert abs(score-result['scores'][split])<1e-12 and abs(score-packet['records'][seed][split])<1e-12
  primary=np.load(P/f'evidence/l131/paper/seed-{seed}/predictions.npz')
  for suffix in ['entity','time','target']:np.testing.assert_array_equal(a[split+'_'+suffix],primary[split+'_'+suffix])
  assert result['replay'][split]['max_original_logit_error']<1e-5
  maxerr=max(maxerr,result['replay'][split]['max_original_logit_error']);count+=n
 t=traces[seed];assert t['status']=='PASS' and t['batch_size']==512
 assert t['parity']['max_output_error']<1e-5 and t['parity']['max_gradient_error']<1e-6
 assert t['parity']['nonfinite_masks']=='MATCH' and t['gradients']['encoder']['nonfinite']==640
 assert t['detached_gradients']['encoder']['with_gradient']==0
b=json.loads((P/'_budget_l131.json').read_text());assert sum(x['workers'] for x in b['reservations'])==10
b['successful_worker_resource_usd']=b['recorded_worker_resource_usd']+r['resource_usd']
b['failed_and_preempted_worker_bound_usd']=3*.812592
b['resource_usd_including_failed_worker_bound']=b['successful_worker_resource_usd']+b['failed_and_preempted_worker_bound_usd']
assert b['resource_usd_including_failed_worker_bound']+b['overhead_reserve_usd']<10
b['status']='COMPLETE_NO_MORE_DISPATCH';b['notebook_worker_resource_usd']=r['resource_usd']
(P/'_budget_l131.json').write_text(json.dumps(b,indent=2)+'\n')
out=dict(status='PASS',code_sha256=r['code_sha256'],code_cells=r['code_cells'],full_gate='FIVE_FULL_FRESH_FITS',all_epochs='COMPLETE',predictions=count,trace_count=5,maximum_original_output_error=maxerr,worker_resource_usd=r['resource_usd'],prior_attempt='PREEMPTED; preserved separately, never merged',live_colab='NOT_CHECKED')
(P/'_notebook_gpu_l131_results.json').write_text(json.dumps(out,indent=2));print(out)
