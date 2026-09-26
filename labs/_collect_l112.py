"""Collect existing experiment evidence; this script never launches training."""
import argparse,hashlib,json
from pathlib import Path
import modal
P=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
s=json.loads((P/'_sources_l112.json').read_text());v=modal.Volume.from_name(s['volume']);sha=s['source_sha256'];missing=[];files={}
for preset,seeds in [('pilot',[100]),('paper',range(10))]:
 for seed in seeds:
  for name in ['identity.json','completed.json','result.json','predictions.npz','checkpoint.pt']+(['source_parity.json'] if preset=='pilot' else []):
   target=(P/'results/l112'/preset/f'seed-{seed}'/name) if name=='checkpoint.pt' else (P/'evidence/l112'/preset/f'seed-{seed}'/name)
   remote=f'/{sha}/{preset}/seed-{seed}/{name}'
   if not target.exists():
    target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix(target.suffix+'.partial')
    try:
     with temp.open('wb') as f:
      for chunk in v.read_file(remote):f.write(chunk)
     temp.replace(target)
    except FileNotFoundError:temp.unlink(missing_ok=True);missing.append(remote);continue
   files[str(target.relative_to(P))]=hashlib.sha256(target.read_bytes()).hexdigest()
(P/'evidence/l112/manifest.json').write_text(json.dumps({'source_sha256':sha,'files':files,'missing':missing},indent=2));print({'files':len(files),'missing':len(missing)})
if missing and not a.partial:raise SystemExit('Incomplete evidence')
