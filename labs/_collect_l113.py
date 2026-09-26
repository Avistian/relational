"""Read evidence only; never launches paid training."""
import hashlib,json
from pathlib import Path
import modal
P=Path(__file__).resolve().parent;s=json.loads((P/'_sources_l113.json').read_text());v=modal.Volume.from_name(s['volume']);sha=s['source_sha256'];files={};missing=[]
def fetch(remote,target):
 target.parent.mkdir(parents=True,exist_ok=True)
 try:
  # These are closed, completed artifacts in an immutable run namespace.
  content=target.read_bytes() if target.exists() else b''.join(v.read_file(remote))
  if not target.exists():target.write_bytes(content)
  files[str(target.relative_to(P))]=hashlib.sha256(content).hexdigest()
 except FileNotFoundError:missing.append(remote)
for name in ['prepared.json','audit.json','labels-splits.npz']:fetch('/data/'+name,P/'evidence/l113'/name)
for preset,seeds in [('pilot',[100]),('paper',range(10))]:
 for seed in seeds:
  for name in ['identity.json','result.json','predictions.npz','checkpoint.pt','replay.json']:
   target=(P/'results/l113' if name=='checkpoint.pt' else P/'evidence/l113')/preset/f'seed-{seed}'/name
   fetch(f'/{sha}/{preset}/seed-{seed}/{name}',target)
(P/'evidence/l113/manifest.json').write_text(json.dumps({'source_sha256':sha,'files':files,'missing':missing},indent=2));print({'files':len(files),'missing':len(missing)})
