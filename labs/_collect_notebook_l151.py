"""Collect pinned notebook execution, independently rescore its extra fit."""
import hashlib,json
from pathlib import Path
import modal,numpy as np,nbformat
from relkit.trial_l139 import keyed_auc
P=Path(__file__).resolve().parent;E=P/'evidence/l151';v=modal.Volume.from_name('l151-portfolio-evidence')
for name in ['execution.json','cost.json','l151-report.json','l151-portfolio-entry.json','l151-runs/seed-1000/result.json','l151-runs/seed-1000/predictions.npz','l151-runs/seed-1000/source_parity.json']:
 raw=b''.join(v.read_file('notebook/'+name));p=E/'notebook'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
r=json.loads((E/'notebook/execution.json').read_text());n=nbformat.read(P/'solutions/0151-classification-portfolio.ipynb',4);code='\n\n'.join(c.source for c in n.cells if c.cell_type=='code');assert hashlib.sha256(code.encode()).hexdigest()==r['code_sha256']
fit=json.loads((E/'notebook/l151-runs/seed-1000/result.json').read_text());z=np.load(E/'notebook/l151-runs/seed-1000/predictions.npz');truth=np.load(E/'prepared/queries.npz')
assert fit['status']=='COMPLETE' and fit['seed']==1000 and fit['track']=='notebook' and len(fit['history'])==20
assert all(h['queries']==11994 for h in fit['history']);count=0
for split in ['val','test']:
 for field in ['study','time','target']:assert np.array_equal(z[split+'_'+field],truth[split+'_'+field])
 keys=list(zip(z[split+'_study'],z[split+'_time']));score=keyed_auc(keys,z[split+'_target'],keys,z[split+'_pred']);assert abs(score-fit['scores'][split]['roc_auc'])<1e-12;count+=len(keys)
checkpoint=P/'results/l151/notebook-1000.pt'
with checkpoint.open('wb') as fp:
 for block in v.read_file('notebook/l151-runs/seed-1000/selected.pt'):fp.write(block)
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==fit['checkpoint_sha256']
assert fit['original_model_parity']['status']=='NUMERIC_CLOSE'
r.update(extra_checkpoint_sha256=fit['checkpoint_sha256'],extra_verified_predictions=count,extra_scores=fit['scores'],excluded_from_primary=True)
(P/'_notebook_l151_results.json').write_text(json.dumps(r,indent=2));print(r)
