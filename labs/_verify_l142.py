"""Final source, primary evidence, notebook and aggregate budget reconciliation."""
import ast,hashlib,json
from pathlib import Path
import numpy as np,nbformat
from _collect_l142 import fetch
P=Path(__file__).resolve().parent;E=P/'evidence/l142'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=json.loads((P/'_sources_l142.json').read_text())
for name,info in source['files'].items():assert sha(P/'sources/l141'/name)==info['sha256']
summary=json.loads((E/'training.json').read_text());prepared=json.loads((E/'prepared/prepared.json').read_text())
budget=json.loads((P/'_budget_l142.json').read_text())
for row in budget['reservations']:
 for name,digest in row['sources'].items():assert sha(P/name)==digest,(row['phase'],name,'source drift')
errors={};nan_counts={}
for arm in ['composite','ordinary']:
 for seed in range(5):
  name=f'{arm}-{seed}';r=json.loads((E/name/'result.json').read_text())
  assert r['graph_sha256']==prepared['graph_sha256']
  assert r['parameter_count']==summary['parameter_counts'][arm]
  assert r['seed']==seed and r['arm']==arm
  z=np.load(E/name/'predictions.npz')
  for split in ['val','test']:assert np.isfinite(z[split+'_pred']).all()
  errors[name]=max(x['max_original_error'] for x in r['scores'].values())
  nan_counts[name]=sum(r['first_batch_nonfinite_gradients'].values())
for relative,digest in summary['source_artifact_hashes'].items():assert sha(E/relative)==digest
# Validate extra full runs executed directly from the portable notebook namespace.
for name in ['notebook/execution.json','notebook/cost.json','notebook/l142-report.json']:fetch(name)
execution=json.loads((E/'notebook/execution.json').read_text())
nb=nbformat.read(P/'solutions/0142-many-to-many-edge-pathology.ipynb',4)
code='\n\n'.join(c.source for c in nb.cells if c.cell_type=='code');assert hashlib.sha256(code.encode()).hexdigest()==execution['code_sha256']
notebook_predictions=0;notebook_scores={}
for arm in ['composite','ordinary']:
 phase=f'notebook/l142-runs/{arm}-100'
 for name in ['result.json','predictions.npz','sampled_trace.npz']:fetch(phase+'/'+name)
 r=json.loads((E/phase/'result.json').read_text());z=np.load(E/phase/'predictions.npz');ref=np.load(E/f'{arm}-0/predictions.npz')
 assert r['seed']==100 and r['epochs']==10 and len(r['history'])==10 and r['arm']==arm
 assert all(x['queries']==7453 and x['steps']==15 for x in r['history'])
 assert r['best_epoch']==1+int(np.argmin([h['val_mae'] for h in r['history']]))
 assert r['temporal_audit']['future_violations']==0 and r['graph_sha256']==prepared['graph_sha256']
 for split in ['val','test']:
  for key in ['entity','time','target']:np.testing.assert_array_equal(z[f'{split}_{key}'],ref[f'{split}_{key}'])
  assert abs(np.mean(np.abs(z[split+'_target']-z[split+'_pred']))-r['scores'][split]['mae'])<1e-10
  notebook_predictions+=len(z[split+'_target'])
 notebook_scores[arm]=r['scores']
 dest=P/'results/l142'/f'notebook-{arm}-100.pt'
 if not dest.exists():
  from _collect_l142 import v
  dest.write_bytes(b''.join(v.read_file(phase+'/selected.pt')))
 assert sha(dest)==r['checkpoint_sha256']
notebook_audit=dict(status='PASS',code_sha256=execution['code_sha256'],cells=execution['cells'],extra_full_fits=2,independently_scored_predictions=notebook_predictions,scores=notebook_scores,pooled_into_primary=False,live_colab='NOT_CHECKED')
(E/'notebook-audit.json').write_text(json.dumps(notebook_audit,indent=2))
body=sum(json.loads(p.read_text())['worker_body_usd'] for p in E.glob('*-cost.json'))+json.loads((E/'notebook/cost.json').read_text())['worker_body_usd']
ceiling=sum(r['upper_usd'] for r in budget['reservations'])+budget['overhead_reserve_usd'];assert ceiling<=budget['budget_usd']
budget['accounting']=dict(conservative_reserved_plus_overhead_usd=ceiling,recorded_worker_body_usd=body,scope='Worker body only; startup/build/commit/storage not itemized, overhead reserved; invoice NOT_ITEMIZED')
(P/'_budget_l142.json').write_text(json.dumps(budget,indent=2))
report=dict(status='PASS',source_hashes='PASS',primary_runs=10,primary_predictions=12590,notebook=notebook_audit,max_oracle_error=max(errors.values()),per_seed_nonfinite_gradients=nan_counts,aggregate_reserved_usd=ceiling,worker_body_estimate_usd=body,historical_identity='NOT_ESTABLISHED',whole_paper='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')
(P/'_verify_l142_results.json').write_text(json.dumps(report,indent=2));print(report)
