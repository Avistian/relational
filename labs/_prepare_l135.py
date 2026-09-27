"""Freeze source identities before dispatch; refuse to overwrite a budget."""
import hashlib,json
from pathlib import Path
P=Path(__file__).parent;R=P.parent
assert not (P/'_budget_l135.json').exists()
files=[P/x for x in ['relkit/rdl_l117.py','relkit/batch_audit_l123.py','relkit/tuning_l135.py','relkit/tuning_train_l135.py','_run_l117.py','_run_l135.py','_protocol_l135.json','requirements-l117-runtime.txt']]+[R/'modal/l135_repro.py']+list((P/'sources/l117').rglob('*'))
hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file() and '__pycache__' not in p.parts}
source=dict(upstream_commit='9aa346267c2e1c560bd92da07d6f4ad1ca2f0639',files=hashes,identity='Released implementation; exact historical training commit NOT_ESTABLISHED',paper_url='https://arxiv.org/html/2407.20060v1')
paper=P/'sources/l135/paper.html'
if paper.exists():source['paper']=dict(url=source['paper_url'],sha256=hashlib.sha256(paper.read_bytes()).hexdigest(),path='labs/sources/l135/paper.html')
(P/'_sources_l135.json').write_text(json.dumps(source,indent=2))
b=dict(budget_usd=10,rate_usd_second=.00022572,timeout_seconds=900,overhead_reserve_usd=3,source_hashes=hashes,reservations=[],pilot_passed=False)
(P/'_budget_l135.json').write_text(json.dumps(b,indent=2));print('Frozen',len(hashes),'source files')
