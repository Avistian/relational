"""Collect artifacts only after worker volume commit; independently score outputs."""
from pathlib import Path
import json,sys
import modal,numpy as np
P=Path(__file__).resolve().parent;E=P/'evidence/l146';v=modal.Volume.from_name('l146-comparison-evidence')
def fetch(name):
 p=E/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b''.join(v.read_file(name)));return p
phase=sys.argv[1]
for suffix in ['-started.json','-cost.json']:fetch(phase+suffix)
if phase=='prepare':
 a=json.loads(fetch('prepared/audit.json').read_text())
 for s in ['train','val','test']:
  z=np.load(fetch('prepared/'+s+'.npz'));assert not ((z['token_times']!=-1)&(z['token_times']>z['cutoff'][:,None])).any()
 print(a)
else:
 try:r=json.loads(fetch(phase+'/result.json').read_text())
 except Exception:
  fetch(phase+'-failure.txt');raise
 z=np.load(fetch(phase+'/predictions.npz'));fetch(phase+'/history.json')
 for split in r['scores']:
  expected=np.load(E/f'prepared/{split}.npz');lookup={(int(i),int(t)):float(y) for i,t,y in zip(expected['entity'],expected['cutoff'],expected['target'])}
  keys=list(zip(z[split+'_entity'],z[split+'_cutoff']));assert len(set(keys))==len(keys)
  assert all(lookup[(int(i),int(t))]==y for (i,t),y in zip(keys,z[split+'_target']))
  if not r['pilot']:assert set(keys)==set(lookup)
  assert abs(float(np.mean(np.abs(z[split+'_target']-z[split+'_pred'])))-r['scores'][split])<1e-10
 print({k:r[k] for k in ['arm','seed','pilot','scores','seconds','parameters']})
