"""Freeze original and fresh evidence; do not silently bless changed packets."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l169'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert (E/'remaining-complete/receipt.json').exists(),'Assemble and inspect complete evidence first'
old=json.loads((P/'evidence/l168/audit-manifest.json').read_text());files=dict(old['files'])
for name,digest in files.items():assert sha(P/name)==digest,name
paths=[P/'evidence/l168/audit-manifest.json',E/'input-manifest.json',E/'rel-f1.npz',E/'rel-trial.npz']
paths+=list((P/'sources/l169').iterdir())
for phase in ['pilot-1','remaining-complete','remaining-2','tail-1']:paths += [p for p in (E/phase).iterdir() if p.suffix in ['.json','.npz']]
for p in paths:files[str(p.relative_to(P))]=sha(p)
packet=dict(files=dict(sorted(files.items())),experiment='L169 complete two-task five-context sweep')
path=E/'audit-manifest.json'
if path.exists():assert json.loads(path.read_text())==packet,'Existing frozen evidence changed'
else:path.write_text(json.dumps(packet,indent=2)+'\n')
b=json.loads((E/'budget.json').read_text());seconds=sum(json.loads((E/phase/'cost.json').read_text())['worker_body_seconds'] for phase in ['pilot-1','remaining-2','tail-1'])
partial_cost=E/'remaining-1/cost.json'
if partial_cost.exists():seconds+=json.loads(partial_cost.read_text())['worker_body_seconds']
cost=dict(cap_usd=10,overhead_reserved_usd=3,total_reserved_usd=3+sum(r['upper_usd'] for r in b['reservations']),known_worker_body_seconds=seconds,known_worker_body_usd=seconds*.00028372,
 failed_attempt='remaining-1 partial; excluded from results; second wait cancelled;225saved runs plus9tail runs;4RDBsentinels exact;2TabICLsentinels show small numerical drift',
 incomplete_attempt_cost='Both cancellation cost receipts included; reservations retained in full',
 invoice='NOT_ITEMIZED',fresh_runs=240,reused_runs=60,inference_retries=2,collection_note='Modal volume download retains remote basename; downloaded nested files moved locally without rerunning compute')
(E/'cost.json').write_text(json.dumps(cost,indent=2)+'\n');print(cost)
