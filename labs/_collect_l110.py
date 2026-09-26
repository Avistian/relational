"""Read existing Modal evidence only; never submits compute or retrains a seed."""
import argparse,hashlib,json
from pathlib import Path
import modal
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');p.add_argument('--checkpoints',action='store_true');a=p.parse_args()
sources=json.loads((P/'_sources_l110.json').read_text());sha=sources['implementation_sha256'];v=modal.Volume.from_name(sources['volume'])
compact=P/'evidence/l110';large=P/'results/l110/checkpoints';compact.mkdir(parents=True,exist_ok=True)
missing=[];manifest={}
def fetch(remote,target,optional=False):
 if target.exists():return True
 target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix(target.suffix+'.partial')
 try:
  with temp.open('wb') as f:
   for chunk in v.read_file(remote):f.write(chunk)
  temp.replace(target);return True
 except FileNotFoundError:
  temp.unlink(missing_ok=True)
  if not optional:missing.append(remote)
  return False
for seed in range(10):
 base=f'/{sha}/paper/seed-{seed}';dest=compact/f'seed-{seed}'
 fetch(base+'/identity.json',dest/'identity.json');fetch(base+'/completed.json',dest/'completed.json')
 for arm in ['release','clean']:
  for suffix in ['.json','-predictions.npz']:
   name=f'seed-{seed}'+suffix;fetch(base+'/'+arm+'/'+name,dest/arm/name)
  if a.checkpoints:
   name=f'seed-{seed}.pt';fetch(base+'/'+arm+'/'+name,large/f'seed-{seed}'/arm/name)
 fetch(base+'/release/seed-0-source-replay.json',dest/'release/seed-0-source-replay.json',optional=True) if seed==0 else None
for name in ['identity.json','completed.json','source_parity.json']:
 fetch(f'/{sha}/pilot/seed-0/'+name,compact/'pilot'/name)
for path in sorted(compact.rglob('*')):
 if path.is_file() and path.name!='manifest.json':manifest[str(path.relative_to(compact))]=hashlib.sha256(path.read_bytes()).hexdigest()
for path in sorted(large.rglob('*.pt')):
 manifest['LOCAL_CHECKPOINT:'+str(path.relative_to(large))]=hashlib.sha256(path.read_bytes()).hexdigest()
(compact/'manifest.json').write_text(json.dumps({'source_sha256':sha,'files':manifest,'missing_at_collection':missing},indent=2))
print({'collected_files':len(manifest),'missing':len(missing)})
if missing and not a.partial:raise SystemExit('Incomplete evidence: run --partial while training, or collect after completion')
