"""Collect actual notebook-gate execution and independently score its extra runs."""
from pathlib import Path
import json,hashlib
import modal,numpy as np
from relkit.relgnn_l143 import keyed_mae
P=Path(__file__).resolve().parent;E=P/'evidence/l143';v=modal.Volume.from_name('l143-relgnn-evidence')
for name in ['notebook/execution.json','notebook/cost.json']:
 p=E/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b''.join(v.read_file(name)))
report=json.loads((E/'notebook/execution.json').read_text());count=0
for lane in ['replay-compatible','seed-100']:
 prefix='notebook/l143-runs/'+lane
 for file in ['result.json','predictions.npz']:
  p=E/prefix/file;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b''.join(v.read_file(prefix+'/'+file)))
 z=np.load(E/prefix/'predictions.npz');r=json.loads((E/prefix/'result.json').read_text());assert r['status']=='COMPLETE'
 base=np.load(E/'seed-0/predictions.npz')
 for split in ['val','test']:
  for name in ['entity','time','target']:assert np.array_equal(z[split+'_'+name],base[split+'_'+name])
  keys=list(zip(z[split+'_entity'],z[split+'_time']))
  score=keyed_mae(keys,z[split+'_target'],keys,z[split+'_pred']);assert abs(score-r['scores'][split]['mae'])<1e-12;count+=len(keys)
 if lane=='seed-100':assert r['seed']==100 and r['epochs']==10 and len(r['history'])==10
report['additional_predictions_independently_scored']=count
(P/'_notebook_l143_results.json').write_text(json.dumps(report,indent=2));print(report)
