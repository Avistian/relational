"""Actual complete checkpoints on a fixed original blobs context and query grid."""
import json,hashlib
from pathlib import Path
import numpy as np,torch
from relkit import driftpfn_l068_v2 as c
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 torch.set_num_threads(1);data=c.load_dataset(ROOT,'blobs');train=np.flatnonzero(data['c']<4)
 xx,yy=np.meshgrid(np.linspace(-20,20,36),np.linspace(-26,17,36));grid=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float32)
 rows=[]
 for arm,variant in [('base_time','base'),('drift','dist')]:
  model,meta=c.load_pretrained(ROOT,variant,1)
  for domain in [4,8,13]:
   x=np.concatenate([data['x'][train],grid]);t=np.concatenate([data['c'][train],np.full(len(grid),domain)]);y=np.concatenate([data['y'][train],np.zeros(len(grid),dtype=int)])
   p=c.predict(model,x,y,t,np.arange(len(train)),np.arange(len(train),len(x)),arm,17)
   rows.append(dict(arm=arm,domain=domain,probabilities=p.tolist(),mean_confidence=float(p.max(1).mean()),checkpoint=meta['checkpoint']))
 r=dict(scope='Original Intersecting Blobs source; fixed domains0..3, checkpoint1, raw numeric one-view wrapper. Different from published optimized Figure5.',grid=grid.tolist(),shape=list(xx.shape),context_ids=train.tolist(),records=rows,kernel_identity=c.kernel_identity(vars(c)),operator_sha256=hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest())
 (ROOT/'_boundary_l068_v2_results.json').write_text(json.dumps(r,indent=2));print([(r['arm'],r['domain'],r['mean_confidence']) for r in rows])
