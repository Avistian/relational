"""Collect independently rescored evidence from portable full-training execution."""
import hashlib,io,json,math
from pathlib import Path
import modal,numpy as np
P=Path(__file__).resolve().parent;v=modal.Volume.from_name('l136-notebook-validation-v2')
def read(path):return b''.join(v.read_file(path))
report=json.loads(read('report.json'));assert report['status']=='PASS'
notebook=json.loads((P/'solutions/0136-leaderboard-literacy.ipynb').read_text());code='\n\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code');assert hashlib.sha256(code.encode()).hexdigest()==report['code_sha256']
count=0;files={};root=P/'evidence/l136/notebook-validation';root.mkdir(parents=True,exist_ok=True)
for seed in range(5):
 target=root/f'seed-{seed}';target.mkdir(exist_ok=True)
 for name in ['result.json','predictions.npz']:
  raw=read(f'work/l136-full/seed-{seed}/'+name);(target/name).write_bytes(raw);files[f'seed-{seed}/'+name]=hashlib.sha256(raw).hexdigest()
 r=json.loads((target/'result.json').read_text());a=np.load(target/'predictions.npz');reference=np.load(P/'evidence/l136/final/lr005-full/seed-0/predictions.npz')
 assert len(r['trace'])==10 and all(t['train_queries']==7453 for t in r['trace'])
 assert r['selected_epoch']==min(r['trace'],key=lambda t:t['val_mae'])['epoch']
 for split in ['val','test']:
  for field in ['target','entity','time']:np.testing.assert_array_equal(a[split+'_'+field],reference[split+'_'+field])
  measured=math.fsum(abs(float(x)-float(y)) for x,y in zip(a[split+'_pred'],a[split+'_target']))/len(a[split+'_pred'])
  assert abs(measured-r['scores'][split])<1e-12
  assert abs(measured-report['packet']['records'][seed]['scores'][split])<1e-12
  count+=len(a[split+'_pred'])
report.update(independent_rescore=dict(status='PASS',predictions=count,query_identity='MATCH_RELEASE_ARCHIVE_CHECKED_PRIMARY'),artifact_hashes=files)
(P/'_notebook_gpu_l136_results.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k not in ['packet','artifact_hashes']})
