"""Freeze training identities; never erase prior reservations."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
assert not (P/'_budget_l136.json').exists()
files=[P/x for x in ['_run_l117.py','_run_l135.py','_run_l136.py','requirements-l117-runtime.txt','relkit/rdl_l117.py','relkit/tuning_train_l135.py','relkit/tuning_l135.py','relkit/batch_audit_l123.py']]+[R/'modal/l136_repro.py']+list((P/'sources/l117').rglob('*'))
hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file() and '__pycache__' not in p.parts}
b=dict(budget_usd=10,rate_usd_second=.00022572,timeout_seconds=900,overhead_reserve_usd=3,source_hashes=hashes,reservations=[],pilot_passed=False)
(P/'_budget_l136.json').write_text(json.dumps(b,indent=2));print('Frozen',len(hashes),'training source identities')
