"""Execute every frozen course arm. No silent overwrite or partial success."""
import json,hashlib
from pathlib import Path
import numpy as np
from relkit.limix_b08 import train_arm
P=Path(__file__).resolve().parent;E=P/'evidence/b08';protocol=json.loads((E/'course-protocol.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P/'relkit/limix_b08.py')==protocol['code_sha256']
for n,h in protocol['data_files'].items():assert sha(P/n)==h,n
out=E/'runs';out.mkdir(exist_ok=True);records=[]
for seed in protocol['seeds']:
 train=dict(np.load(P/f'data/b08/train-{seed}.npz'));test=dict(np.load(P/f'data/b08/test-{seed}.npz'))
 for objective in protocol['objectives']:
  f=out/f'{objective}-{seed}.npz';record=f.with_suffix('.json')
  if f.exists() or record.exists():raise SystemExit('Refusing to overwrite '+str(f))
  arrays,r=train_arm(train,test,seed,objective)
  np.savez_compressed(f,**arrays);r['prediction_sha256']=sha(f);r['file']=f.name
  record.write_text(json.dumps(r,indent=2)+'\n');records.append(r)
  print(objective,seed,round(r['seconds'],2),'seconds',flush=True)
(out/'results.json').write_text(json.dumps(dict(protocol_sha256=sha(E/'course-protocol.json'),records=records),indent=2)+'\n')
