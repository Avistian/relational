"""Collect and independently audit the separate full-notebook validation lane."""
from pathlib import Path
import hashlib,json
import modal,numpy as np,nbformat
from sklearn.metrics import roc_auc_score
P=Path(__file__).resolve().parent;E=P/'evidence/l139';v=modal.Volume.from_name('l139-full-notebook')
def fetch(remote,local):
 raw=b''.join(v.read_file(remote));local.parent.mkdir(parents=True,exist_ok=True);local.write_bytes(raw);return raw
r=json.loads(fetch('result.json',P/'_notebook_full_l139_results.json'));assert r['status']=='PASS'
notebook=nbformat.read(P/'solutions/0139-healthcare-trial.ipynb',4);digest=hashlib.sha256('\n\n'.join(c.source for c in notebook.cells if c.cell_type=='code').encode()).hexdigest();assert digest==r['code_sha256']
assert r['preparation']['source_sha256']==hashlib.sha256((P/'_full_l139.py').read_bytes()).hexdigest()
fetch('prepare-cost.json',E/'fresh-preparation-cost.json');fetch('cost.json',E/'fresh-notebook-cost.json')
queries=np.load(E/'queries.npz');scores={};n=0;hashes={}
for seed in range(5):
 for name in ['result.json','predictions.npz','data_identity.json','progress.json','source_parity.json','sampled_trace.npz']:
  dest=E/'notebook-validation'/f'seed-{seed}'/name;raw=fetch(f'l139-full/seed-{seed}/{name}',dest);hashes[str(dest.relative_to(E))]=hashlib.sha256(raw).hexdigest()
 path=E/'notebook-validation'/f'seed-{seed}';result=json.loads((path/'result.json').read_text());assert result['seed']==seed and result['epochs']==20 and len(result['history'])==20
 assert result['best_epoch']==1+int(np.argmax([x['val']['roc_auc'] for x in result['history']]))
 assert result['original_model_parity']['status']=='NUMERIC_CLOSE'
 assert result['temporal_audit']['query_occurrences']==260865 and result['temporal_audit']['future_violations']==0
 data=np.load(path/'predictions.npz')
 for split in ['val','test']:
  for field in ['study','time','target']:assert np.array_equal(data[split+'_'+field],queries[split+'_'+field])
  score=roc_auc_score(queries[split+'_target'],data[split+'_pred']);assert abs(score-result['scores'][split]['roc_auc'])<1e-12;scores[f'{seed}-{split}']=score;n+=len(data[split+'_pred'])
summary=dict(status='PASS',scope='Separate notebook execution validation; not pooled with primary results',verified_predictions=n,scores=scores,artifact_hashes=hashes,mean_auc={split:float(np.mean([scores[f'{seed}-{split}'] for seed in range(5)])) for split in ['val','test']},code_sha256=digest,fresh_graph='EXACT SHA256 match to primary materialization',historical_identity='NOT_ESTABLISHED')
(P/'_notebook_full_audit_l139_results.json').write_text(json.dumps(summary,indent=2));print({k:summary[k] for k in ['status','verified_predictions','mean_auc']})
