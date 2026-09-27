"""Collect and independently validate compact evidence from full portable execution."""
import hashlib,io,json,math
from pathlib import Path
import modal,numpy as np
from relkit.tuning_l135 import select_configuration,paired_differences
P=Path(__file__).parent;v=modal.Volume.from_name('l135-notebook-validation')
def read(path):return b''.join(v.read_file(path))
report=json.loads(read('report.json'));assert report['status']=='PASS'
notebook=json.loads((P/'solutions/0135-tuning-on-reg.ipynb').read_text());code='\n\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
assert hashlib.sha256(code.encode()).hexdigest()==report['code_sha256']
protocol=json.loads((P/'_protocol_l135.json').read_text());rows=[];final={};count=0;files={}
root=P/'sources/l135/notebook-validation';root.mkdir(parents=True,exist_ok=True)
for phase in ['search','final']:
 configs=[c['id'] for c in protocol['configurations']] if phase=='search' else list(dict.fromkeys([protocol['default'],report['packet']['winner']]))
 seeds=protocol['search_seeds'] if phase=='search' else protocol['final_seeds']
 for config in configs:
  for seed in seeds:
   stem=f'{phase}/{config}/seed-{seed}';target=root/stem;target.mkdir(parents=True,exist_ok=True)
   for name in ['result.json','predictions.npz']:
    raw=(target/name).read_bytes() if (target/name).exists() else read('work/l135-full/'+stem+'/'+name);(target/name).write_bytes(raw);files[stem+'/'+name]=hashlib.sha256(raw).hexdigest()
   r=json.loads((target/'result.json').read_text());a=np.load(target/'predictions.npz')
   assert len(r['trace'])==10 and all(t['train_queries']==7453 for t in r['trace'])
   assert r['selected_epoch']==min(r['trace'],key=lambda t:t['val_mae'])['epoch']
   assert set(r['scores'])==({'val'} if phase=='search' else {'val','test'})
   for split,score in r['scores'].items():
    reference=np.load(P/f'evidence/l135/final/lr005-full/seed-0/predictions.npz')
    for field in ['target','entity','time']:np.testing.assert_array_equal(a[split+'_'+field],reference[split+'_'+field])
    n=len(a[split+'_pred']);actual=math.fsum(abs(float(x)-float(y)) for x,y in zip(a[split+'_pred'],a[split+'_target']))/n
    assert abs(actual-score)<1e-12;count+=n
   if phase=='search':rows.append(dict(config=config,seed=seed,selection_mae=r['selection_mae']))
   else:final.setdefault(config,[]).append(dict(seed=seed,mae=r['scores']['test']))
selected=select_configuration(rows,[c['id'] for c in protocol['configurations']],protocol['search_seeds'])
assert selected['winner']==report['packet']['winner']
paired=paired_differences(final[protocol['default']],final[selected['winner']])
assert paired['seeds']==report['packet']['paired']['seeds']
assert all(abs(a-b)<1e-12 for a,b in zip(paired['differences'],report['packet']['paired']['differences']))
for key in ['mean_difference','sample_sd']:assert abs(paired[key]-report['packet']['paired'][key])<1e-12
report.update(independent_rescore=dict(status='PASS',predictions=count,query_identity='MATCH_RELEASE_ARCHIVE_CHECKED_PRIMARY',winner=selected['winner']),artifact_hashes=files)
(P/'_notebook_gpu_l135_results.json').write_text(json.dumps(report,indent=2))
bp=P/'_budget_l135.json';b=json.loads(bp.read_text());primary=json.loads((P/'evidence/l135/summary.json').read_text())
b.update(status='COMPLETE',measured_primary_worker_resource_usd=primary['worker_resource_usd'],measured_notebook_worker_resource_usd=report['resource_usd'],measured_total_worker_resource_usd=primary['worker_resource_usd']+report['resource_usd'],billing_total='NOT_ITEMIZED')
bp.write_text(json.dumps(b,indent=2));print({k:v for k,v in report.items() if k not in ['artifact_hashes','packet']})
