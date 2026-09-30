"""Collect and independently validate the extra standalone notebook fit."""
import json,hashlib
from pathlib import Path
import nbformat,numpy as np
from sklearn.metrics import roc_auc_score
from _collect_l140 import fetch
P=Path(__file__).resolve().parent;E=P/'evidence/l140'
for name in ['notebook-trial.json','notebook-trial-cost.json','notebook-trial/result.json','notebook-trial/predictions.npz','notebook-trial/source_parity.json']:fetch(name)
r=json.loads((E/'notebook-trial.json').read_text());nb=nbformat.read(P/'solutions/0140-rdl-reproduction-checkpoint.ipynb',4)
digest=hashlib.sha256('\n\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest();assert r['code_sha256']==digest and r['status']=='PASS'
result=json.loads((E/'notebook-trial/result.json').read_text());assert result['seed']==100 and result['epochs']==20 and len(result['history'])==20
assert result['best_epoch']==1+int(np.argmax([x['val']['roc_auc'] for x in result['history']]))
z=np.load(E/'notebook-trial/predictions.npz');base=np.load(P/'evidence/l139/queries.npz');count=0
for split in ['val','test']:
 for name,old in [('entity','study'),('time','time'),('target','target')]:assert np.array_equal(z[split+'_'+name],base[split+'_'+old])
 assert abs(roc_auc_score(base[split+'_target'],z[split+'_pred'])-result['scores'][split]['roc_auc'])<1e-12
 count+=len(z[split+'_pred'])
assert result['original_model_parity']['status']=='NUMERIC_CLOSE'
r.update(independently_rescored_extra_predictions=count,primary_mean_inclusion=False)
(P/'_notebook_trial_l140_results.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k!='report'})
