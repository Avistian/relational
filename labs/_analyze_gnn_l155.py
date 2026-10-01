"""Rescore complete new evidence and validate every seed, checkpoint and bin."""
import hashlib,json,statistics
from pathlib import Path
import numpy as np
from relkit.regression_l152 import keyed_metrics,median_diagnostics,portfolio_summary
P=Path(__file__).resolve().parent;E=P/'evidence/l155';records=[];bins={};hashes={};count=0;occurrences=0;gradient={};cost=0;maximum=0
for seed in range(5):
 root=E/f'paper/seed-{seed}';a=np.load(root/'predictions.npz');r=json.loads((root/'result.json').read_text());done=json.loads((root/'completed.json').read_text());d=json.loads((root/'diagnostics.json').read_text());t=json.loads((root/'temporal-audit.json').read_text())
 assert done['status']=='COMPLETE' and r['epochs']==10 and len(r['trace'])==10
 assert all(x['train_queries']==7453 for x in r['trace'])
 assert r['selected_epoch']==min(r['trace'],key=lambda x:x['val_mae'])['epoch']
 metrics={}
 for split in ['val','test']:
  q=[dict(entity=int(e),time=int(t),target=float(y)) for e,t,y in zip(a[split+'_entity'],a[split+'_time'],a[split+'_target'])]
  pred=[dict(entity=int(e),time=int(t),prediction=float(y)) for e,t,y in zip(a[split+'_entity'],a[split+'_time'],a[split+'_pred'])]
  metrics[split]=keyed_metrics(q,list(reversed(pred)));assert abs(metrics[split]['mae']-r['scores'][split])<1e-12
  assert metrics[split]['n']==(499 if split=='val' else 760);count+=metrics[split]['n']
  maximum=max(maximum,r['replay'][split]['max_original_logit_error'])
 edges=np.unique(np.quantile(a['val_pred'],[.2,.4,.6,.8])).tolist();assert edges==d['edges']
 bins[str(seed)]={split:median_diagnostics(a[split+'_target'],a[split+'_pred'],edges) for split in ['val','test']}
 assert bins[str(seed)]==d['bins']
 records.append(dict(seed=seed,epochs=10,complete=True,selected_epoch=r['selected_epoch'],selection_mae=r['selection_mae'],val_mae=metrics['val']['mae'],test_mae=metrics['test']['mae'],test_rmse=metrics['test']['rmse'],test_bias=metrics['test']['bias']))
 occurrences+=sum(x['queries'] for x in t['splits'].values());gradient[str(seed)]=sum(d['first_backward_nonfinite'].values());cost+=json.loads((root/'cost.json').read_text())['worker_body_usd']
 for p in root.iterdir():hashes[str(p.relative_to(P))]=hashlib.sha256(p.read_bytes()).hexdigest()
summary=dict(status='PASS',records=records,metrics=portfolio_summary(records),bins=bins,predictions=count,query_occurrences=occurrences,first_backward_nonfinite=gradient,worker_body_usd=cost,maximum_original_output_error=maximum,hashes=hashes)
(E/'summary.json').write_text(json.dumps(summary,indent=2));print({k:v for k,v in summary.items() if k not in ['hashes','bins','records']})
