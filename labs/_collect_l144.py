"""Collect all small artifacts and independently verify every saved ranking."""
from pathlib import Path
import json,sys
import modal,numpy as np
from relkit.context_l144 import keyed_map
P=Path(__file__).resolve().parent;E=P/'evidence/l144'
def collect():
 v=modal.Volume.from_name('l144-contextgnn-evidence');items=list(v.iterdir('/',recursive=True));files=[]
 for item in items:
  name=item.path.lstrip('/')
  if name.startswith(('cache/','prepared/materialized/')) or not name.endswith(('.json','.npz','.txt')):continue
  dest=E/name;dest.parent.mkdir(parents=True,exist_ok=True)
  dest.write_bytes(b''.join(v.read_file(name)));files.append(name)
 checked=0;results=[]
 for path in sorted(E.glob('pilot-*/result.json')):
  r=json.loads(path.read_text());z=np.load(path.parent/'predictions.npz');keys=list(zip(z['val_entity'],z['val_time']));targets=json.loads((path.parent/'val-targets.json').read_text())
  order=np.arange(len(keys))[::-1];score=keyed_map(keys,targets,[keys[i] for i in order],z['val_pred'][order],z['val_pred'].shape[1]);assert abs(score-r['scores']['val'])<1e-12;checked+=len(keys);results.append(r)
 report={'collected_files':files,'verified_ranking_rows':checked,'pilots':results,'status':'PILOT_ONLY' if results else 'PREPARING','full_selected_reproduction':'INCOMPLETE','historical_identity':'NOT_ESTABLISHED','whole_paper':'NOT_RUN','live_colab':'NOT_CHECKED','learner':'PENDING_WRITTEN_DEFENSE'}
 (E/'summary.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k not in ['pilots','collected_files']})
if __name__=='__main__':collect()
