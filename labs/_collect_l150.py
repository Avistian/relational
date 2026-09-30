"""Collect immutable phases; independently score complete query populations."""
import argparse,json,hashlib,math
from pathlib import Path
import modal,numpy as np
P=Path(__file__).resolve().parent;E=P/'evidence/l150';v=modal.Volume.from_name('l150-relgnn-evidence')
def fetch(name):
 raw=b''.join(v.read_file(name));dest=E/name;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():assert dest.read_bytes()==raw,'Changed remote artifact'
 else:dest.write_bytes(raw)
 return dest

def collect(phase):
 if phase=='prepare':
  for name in ['prepared/prepared.json','prepare-started.json','prepare-cost.json','operator-parity.json']:fetch(name)
  return
 if phase=='diagnosis':fetch('diagnosis.json');return
 phases={'reference':[f'ref-{s}' for s in range(5)],'search':[f'search-{i:03d}' for i in [1,3,5]],'selected':[f'selected-{s}' for s in range(10,15)]}.get(phase,[phase])
 for name in phases:
  for file in ['result.json','predictions.npz','sampled_trace.npz']:fetch(name+'/'+file)
  for file in ['-started.json','-cost.json']:fetch(name+file)
  r=json.loads((E/name/'result.json').read_text());a=np.load(E/name/'predictions.npz');assert r['status']=='COMPLETE'
  splits=['val'] if name.startswith('search') else ['val','test']
  assert set(r['scores'])==set(splits)
  if name.startswith('search'):assert not any(k.startswith('test') for k in a.files) and r['test_access']=='FORBIDDEN'
  for split in splits:
   assert len(a[split+'_target'])==(499 if split=='val' else 760)
   keys=list(zip(a[split+'_entity'],a[split+'_time']));assert len(set(keys))==len(keys)
   score=math.fsum(abs(float(y)-float(p)) for y,p in zip(a[split+'_target'],a[split+'_pred']))/len(keys)
   assert abs(score-r['scores'][split]['mae'])<1e-12
  if name!='replay-compatible':
   assert r['epochs']==10 and len(r['history'])==10
   assert r['best_epoch']==min(r['history'],key=lambda h:h['val_mae'])['epoch']
   assert all(h['queries']==7453 and h['steps']==15 for h in r['history'])
   path=P/'results/l150'/f'{name}.pt';path.parent.mkdir(parents=True,exist_ok=True)
   with path.open('wb') as fp:
    for block in v.read_file(name+'/selected.pt'):fp.write(block)
   assert hashlib.sha256(path.read_bytes()).hexdigest()==r['checkpoint_sha256']
  else:fetch('checkpoint-incompatibility.txt');fetch('compatible/prepared.json')
  assert r['temporal_audit']['future_violations']==0
  trace=np.load(E/name/'sampled_trace.npz')
  for key in trace.files:
   if key.endswith('_times'):assert (trace[key]<=trace['cutoffs'][trace[key[:-6]+'_owners']]).all()
  print(name,r['scores'],round(r['seconds'],2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase');collect(p.parse_args().phase)
