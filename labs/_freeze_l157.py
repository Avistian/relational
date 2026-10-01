import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent/'releases/l157-f1-audit'
paths=[p for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.txt','.json'] and p.name not in ['budget.json','experiment-manifest.json'] and 'evidence' not in p.parts]
m={str(p.relative_to(D)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
f=D/'experiment-manifest.json'
if f.exists():assert json.loads(f.read_text())==m,'Frozen computation changed'
else:f.write_text(json.dumps(m,indent=2)+'\n')
b=D/'budget.json'
if not b.exists():b.write_text(json.dumps(dict(budget_usd=10,overhead_reserve_usd=3,rate_per_second=.00022572,max_worker_slots=12,reservations=[]),indent=2))
print('Frozen',len(m),'experiment files')
