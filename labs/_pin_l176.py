"""Freeze original/new raw evidence after complete collection; refuse silent resealing."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l176'
pins=json.loads((E/'inherited-manifest.json').read_text());files=dict(pins['files'])
for n,h in files.items():assert hashlib.sha256((P/n).read_bytes()).hexdigest()==h,n
paths=[E/'inherited-manifest.json',E/'input-manifest.json',E/'preflight.json',E/'rel-f1.npz',E/'rel-trial.npz']
for phase,count in [('pilot-1',6),('remaining-1',294)]:
 receipt=json.loads((E/phase/'receipt.json').read_text());assert len(receipt['records'])==count
 paths+=list((E/phase).glob('*.json'))+list((E/phase).glob('*.npz'))
for p in paths:files[p.relative_to(P).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
result=dict(experiment='L176 full original replay and nested-support experiment',files=dict(sorted(files.items())))
path=E/'audit-manifest.json'
if path.exists():assert json.loads(path.read_text())==result,'Frozen evidence changed'
else:path.write_text(json.dumps(result,indent=2)+'\n')
b=json.loads((E/'budget.json').read_text());seconds=sum(json.loads((E/phase/'cost.json').read_text())['worker_body_seconds'] for phase in ['pilot-1','remaining-1'])
cost=dict(cap_usd=10,total_reserved_usd=b['reserved_usd'],overhead_reserved_usd=b['overhead_usd'],worker_body_seconds=seconds,worker_body_estimate_usd=seconds*.00028372,invoice='NOT_ITEMIZED',fresh_runs=300,reused_runs=300,inference_retries=0,notes='Worker estimate excludes unitemized build/startup/storage; full timeout reservations retained. Two detached CLI final-log timeouts did not cancel remote work.')
(E/'cost.json').write_text(json.dumps(cost,indent=2)+'\n');print('Pinned',len(files),'files',cost)
