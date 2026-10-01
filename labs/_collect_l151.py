"""Collect all evidence before aggregation; checkpoint streams retained locally."""
import argparse,json,hashlib
from pathlib import Path
import modal,numpy as np
from sklearn.metrics import roc_auc_score
P=Path(__file__).resolve().parent;E=P/'evidence/l151';v=modal.Volume.from_name('l151-portfolio-evidence')
def fetch(name):
 raw=b''.join(v.read_file(name));dest=E/name;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():assert dest.read_bytes()==raw,'Changed remote artifact'
 else:dest.write_bytes(raw)
 return dest

def collect(phase):
 if phase=='prepare':
  for name in ['prepared/prepared.json','prepared/task_audit.json','prepared/samples.json','prepared/queries.npz','prepared/data_identity.json','prepare-started.json','prepare-cost.json']:fetch(name)
  return
 phases={'reference':[f'ref-{s}' for s in range(5)],'search':[f'search-{i:03d}' for i in [50,100,200]],'selected':[f'selected-{s}' for s in range(10,15)]}.get(phase,[phase])
 for name in phases:
  for file in ['result.json','predictions.npz','sampled_trace.npz','source_parity.json']:fetch(name+'/'+file)
  for file in ['-started.json','-cost.json']:fetch(name+file)
  r=json.loads((E/name/'result.json').read_text());a=np.load(E/name/'predictions.npz');assert r['status']=='COMPLETE'
  splits=['val'] if name.startswith('search') else ['val','test']
  assert set(r['scores'])==set(splits)
  if name.startswith('search'):assert not any(k.startswith('test') for k in a.files) and r['test_access']=='FORBIDDEN'
  truth=np.load(E/'prepared/queries.npz')
  for split in splits:
   assert len(a[split+'_target'])==(960 if split=='val' else 825)
   for field in ['study','time','target']:assert np.array_equal(a[split+'_'+field],truth[split+'_'+field])
   keys=list(zip(a[split+'_study'],a[split+'_time']));assert len(set(keys))==len(keys)
   assert abs(roc_auc_score(a[split+'_target'],a[split+'_pred'])-r['scores'][split]['roc_auc'])<1e-12
  assert r['epochs']==20 and len(r['history'])==20
  assert r['best_epoch']==max(r['history'],key=lambda h:h['val']['roc_auc'])['epoch']
  assert all(h['queries']==11994 and h['steps']==24 for h in r['history'])
  path=P/'results/l151'/f'{name}.pt';path.parent.mkdir(parents=True,exist_ok=True)
  if not path.exists():
   with path.open('wb') as fp:
    for block in v.read_file(name+'/selected.pt'):fp.write(block)
  assert hashlib.sha256(path.read_bytes()).hexdigest()==r['checkpoint_sha256']
  assert r['temporal_audit']['future_violations']==0
  trace=np.load(E/name/'sampled_trace.npz')
  for key in trace.files:
   if key.endswith('_times'):assert (trace[key]<=trace['cutoffs'][trace[key[:-6]+'_owners']]).all()
  print(name,r['scores'],round(r['seconds'],2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase');collect(p.parse_args().phase)
