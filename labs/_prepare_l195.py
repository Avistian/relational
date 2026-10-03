"""Freeze approved evidence with original manifest chains; no model execution."""
import hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l195';Q=E/'packet'
Q.mkdir(parents=True,exist_ok=True);origins={}
def copy(src,dst,expected=None):
 data=src.read_bytes();h=hashlib.sha256(data).hexdigest()
 if expected is not None and h!=expected:raise ValueError('Original manifest mismatch: '+str(src))
 target=Q/dst;target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists() and target.read_bytes()!=data:raise ValueError('Refuse to change frozen input '+dst)
 target.write_bytes(data);origins[dst]=dict(path=str(src.relative_to(R)),sha256=h,original_manifest_checked=expected is not None)
old=P/'evidence/l149';frozen=json.loads((old/'frozen.json').read_text());training=json.loads((old/'training.json').read_text())
for name in ['training.json','errors.json','frozen.json','sources.json','fe/summary.json']:copy(old/name,'l149/'+name)
for arm,base in [('gnn','final/lr005-full'),('fe','fe/paper')]:
 for s in range(5):
  folder=old/base/f'seed-{s}';receipt=json.loads((folder/'result.json').read_text())
  for name in ['predictions.npz','result.json']:
   src=folder/name;rel=str(src.relative_to(R));h=training['files'].get(rel) if arm=='gnn' else (receipt['files']['predictions.npz'] if name=='predictions.npz' else frozen['files'].get(rel))
   if h is None and name=='predictions.npz':raise ValueError('Missing original prediction hash '+rel)
   copy(src,f'l149/{arm}/{s}/{name}',h)
old=P/'evidence/l190';m=json.loads((old/'input-manifest.json').read_text())
copy(old/'input-manifest.json','l182/l190-input-manifest.json')
for name,h in m['files'].items():
 if name.startswith('model/') or name=='reports/l182.json':copy(old/'packet'/name,'l182/'+name,h)
receipt=json.loads((Q/'l182/model/full-1/receipt.json').read_text())
copy(P/'evidence/l182/input-manifest.json','l182/original-input-manifest.json',receipt['input_manifest_sha256'])
old=P/'evidence/l194';m=json.loads((old/'input-manifest.json').read_text());art=json.loads((old/'artifact-manifest.json').read_text())
copy(old/'input-manifest.json','l194/input-manifest.json')
copy(old/'report.json','l194/report.json',art['files']['labs/evidence/l194/report.json'])
for name,h in m['files'].items():copy(old/'packet'/name,'l194/packet/'+name,h)
for n in [149,182,190,194]:copy(P/f'l{n}-reproduction.md',f'contracts/l{n}.md')
(Q/'origins.json').write_text(json.dumps(origins,indent=2)+'\n')
manifest=dict(experiment='L195 complete thesis stress-test replay',files={str(p.relative_to(Q)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Q.rglob('*')) if p.is_file()},cloud_usd=0,local_cap_seconds=1800)
(E/'input-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Frozen',len(manifest['files']),'files')
