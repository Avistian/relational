"""Freeze approved L103 inputs before launching any L104 experiments."""
import hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
sha=lambda path:hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()
root=P/'results/l103/gpu'
manifest={'schema':1,'named_target':'Xu et al. 2020 Tables 1/2: TGAT Wikipedia release evaluation','source_commit':'9293d10d1943c4bd4a186337cf38ba98e4c8bb99','implementation_sha256':sha(P/'relkit/tgat_l103.py'),'raw_sha256':sha(P/'l103-cache/wikipedia.csv'),'processed_arrays':{},'seeds':{}}
with np.load(P/'l103-cache/processed.npz') as z:
 for k in ['u','v','t','x']:manifest['processed_arrays'][k]={'shape':list(z[k].shape),'dtype':str(z[k].dtype),'sha256':hashlib.sha256(z[k].tobytes()).hexdigest()}
for seed in range(10):
 folder=root/f'seed-{seed}'
 if not all((folder/name).exists() for name in ['selected.pt','predictions.npz','identity.json','result.json']):
  print(f'Pending upstream training: seed {seed}');continue
 manifest['seeds'][str(seed)]={name:sha(folder/name) for name in ['selected.pt','predictions.npz','identity.json','result.json']}
 identity=json.loads((folder/'identity.json').read_text())
 assert identity['implementation_sha256']==manifest['implementation_sha256']
 assert identity['data']['raw_sha256']==manifest['raw_sha256']
 if seed==0:manifest['split_sha256']=identity['data']['split_sha256']
 else:assert manifest['split_sha256']==identity['data']['split_sha256']
f=P/'_inputs_l104.json'
if f.exists():
 old=json.loads(f.read_text())
 for key in old:
  if key=='seeds':
   for seed,record in old[key].items():assert manifest[key][seed]==record,'Frozen seed drift'
  else:assert manifest[key]==old[key],'Frozen input drift'
f.write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Frozen {len(manifest["seeds"])} complete checkpoint/prediction identities and raw/processed data')
