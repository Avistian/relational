import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];E=R/'labs/evidence/b14'
lock={}
for root,dest in [(R/'labs/sources/b14/relarena','/source'),(Path('/tmp/b14-dfs-cache'),'/dfs-cache'),(Path.home()/'.cache/relbench/rel-f1','/root/.cache/relbench/rel-f1'),(Path('/tmp/b14-weights'),'/root/.cache/tabpfn')]:
 for p in root.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:lock[dest+'/'+str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
(E/'cloud-input-lock.json').write_text(json.dumps(lock,indent=2)+'\n')
rate=.000694+4*.0000131+24*.00000222
p=E/'cloud-budget.json'
if not p.exists():p.write_text(json.dumps(dict(cap_usd=10,stop_usd=8,reserve_usd=2,overhead_reserve_usd=2,rate_usd_second=rate,rate_source='https://modal.com/pricing',hardware='A100-80GB,4CPU,24GiB',reservations=[]),indent=2)+'\n')
print('Input files',len(lock),'reserved full maximum including overhead',2+rate*(360+1260))
