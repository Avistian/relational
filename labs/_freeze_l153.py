from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent;R=P.parent
path=P/'_budget_l153.json'
files=[P/n for n in ['_prepare_l153.py','_run_l153.py','_labels_l153.py','relkit/recommendation_model_l153.py','relkit/recommendation_l153.py','relkit/batch_audit_l123.py','requirements-l117-runtime.txt']]+sorted((P/'sources/l153').glob('*'))
b=dict(budget_usd=10,overhead_reserve_usd=3,rate_usd_second=.00026124,rates_checked='2026-10-01',rates_url='https://modal.com/pricing',source_hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},reservations=[])
assert not path.exists(),'Do not overwrite immutable reservations'
path.write_text(json.dumps(b,indent=2))
