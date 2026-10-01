"""Collect full-gate evidence, independently rescore and compare source hashes."""
import json,hashlib
from pathlib import Path
import modal,numpy as np,nbformat
from relkit.regression_l152 import keyed_metrics,median_diagnostics,portfolio_summary
P=Path(__file__).resolve().parent;E=P/'evidence/l152/notebook';E.mkdir(parents=True,exist_ok=True);v=modal.Volume.from_name('l152-regression-evidence')
names=['execution.json','cost.json','l152-report.json','l152-own-run-entry.json']+[f'l152-own-runs/seed-{s}/{n}' for s in range(5) for n in ['predictions.npz','result.json','audit.json','temporal-audit.json','checkpoint-audit.json','diagnostics.json']]
for name in names:
 dest=E/name;dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('wb') as f:
  for chunk in v.read_file('notebook/'+name):f.write(chunk)
nb=nbformat.read(P/'solutions/0152-regression-portfolio.ipynb',4);code='\n\n'.join(c.source for c in nb.cells if c.cell_type=='code');report=json.loads((E/'execution.json').read_text());assert hashlib.sha256(code.encode()).hexdigest()==report['code_sha256']
records=[];count=0
for seed in range(5):
 root=E/f'l152-own-runs/seed-{seed}';a=np.load(root/'predictions.npz');r=json.loads((root/'result.json').read_text());assert len(r['trace'])==10 and all(t['train_queries']==7453 for t in r['trace']);assert r['selected_epoch']==min(r['trace'],key=lambda x:x['val_mae'])['epoch'];scores={}
 primary=np.load(P/f'evidence/l152/paper/seed-{seed}/predictions.npz')
 for split in ['val','test']:
  for k in ['entity','time','target']:np.testing.assert_array_equal(a[split+'_'+k],primary[split+'_'+k])
  q=[dict(entity=int(e),time=int(t),target=float(y)) for e,t,y in zip(a[split+'_entity'],a[split+'_time'],a[split+'_target'])]
  p=[dict(entity=int(e),time=int(t),prediction=float(y)) for e,t,y in zip(a[split+'_entity'],a[split+'_time'],a[split+'_pred'])]
  scores[split]=keyed_metrics(q,p);assert abs(scores[split]['mae']-r['scores'][split])<1e-12;count+=len(q)
 d=json.loads((root/'diagnostics.json').read_text());edges=np.unique(np.quantile(a['val_pred'],[.2,.4,.6,.8])).tolist();assert edges==d['edges'];assert all(median_diagnostics(a[s+'_target'],a[s+'_pred'],edges)==d['bins'][s] for s in ['val','test'])
 records.append(dict(seed=seed,epochs=10,complete=True,val_mae=scores['val']['mae'],test_mae=scores['test']['mae'],test_rmse=scores['test']['rmse']))
metrics=portfolio_summary(records);own=json.loads((E/'l152-own-run-entry.json').read_text());assert metrics==own['metrics'] and own['evidence_owner']=='OWN_FRESH_RUN';assert count==6295
report.update(predictions=count,metrics=metrics,cost=json.loads((E/'cost.json').read_text()),live_colab='NOT_CHECKED');(P/'_notebook_l152_results.json').write_text(json.dumps(report,indent=2));print(report)
