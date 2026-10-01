"""Single-use source/config freeze before any paid dispatch."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l156'
paths=['modal/l156_repro.py','labs/_run_l156.py','labs/relkit/temporal_audit_l156.py','labs/_run_l152.py','labs/_run_l117.py','labs/relkit/rdl_l117.py','labs/relkit/batch_audit_l123.py','labs/relkit/regression_l152.py','labs/requirements-l117-runtime.txt','labs/_preflight_l156.py','labs/_audit_fe_l156.py','labs/sources/l129/f1/driver-position/feats.sql','labs/evidence/l156/correction-protocol.json']
paths += [str(p.relative_to(R)) for p in (P/'sources/l117').iterdir() if p.is_file()]
protocol=dict(name='L156 rel-f1/driver-position temporal audit',source_commit='9aa346267c2e1c560bd92da07d6f4ad1ca2f0639',seeds=list(range(5)),epochs=10,selection='first strict minimum validation MAE',paper_targets=dict(val=3.193,test=4.022),mean_tolerance=.2,fit_horizon='2005-01-01',lanes=['released','fit_horizon'],historical_availability='NOT_ESTABLISHED')
(E/'protocol.json').write_text(json.dumps(protocol,indent=2))
paths.append('labs/evidence/l156/protocol.json')
b=dict(budget_usd=10,overhead_reserve_usd=3,max_worker_slots=12,rate_per_second=.00022572,reservations=[],source_hashes={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in paths})
f=P/'_budget_l156.json'
if f.exists():assert json.loads(f.read_text())==b,'Refuse re-freezing an existing ledger'
else:f.write_text(json.dumps(b,indent=2))
(E/'source-manifest.json').write_text(json.dumps(b['source_hashes'],indent=2));print('Frozen',len(paths),'source/protocol files')
