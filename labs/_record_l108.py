"""Record this lesson's verified scope without changing other lessons' entries."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
read=lambda name:json.loads((P/name).read_text())
a=read('_analysis_l108_results.json');b=read('_budget_l108.json')
checks={name:read(f'_{name}_l108_results.json')['status'] for name in ['check','audit','resume','mutation','provenance','execution','delivery']}
assert all(x=='PASS' for x in checks.values()) and a['status']=='COMPLETE'
p=P/'reproductions/execution_evidence.json';ledger=json.loads(p.read_text())
ledger['lesson_108']={'date':'2026-09-26','scope':'Ten existing TGAT Wikipedia checkpoints: complete released evaluation and four fixed-weight sampling interventions; full-data CPU sampler benchmark','status':'COMPLETE','fingerprint':a['fingerprint'],'source_commit':'9293d10d1943c4bd4a186337cf38ba98e4c8bb99','raw_events':157474,'release_ap_percent':a['release'],'release_prediction_max_error':a['release_prediction_max_error'],'paired_intervention_positive_questions':a['intervention_positive_questions'],'checks':checks,'artifacts':'labs/evidence/l108/full/seed-N/{result.json,predictions.npz}','protocol':'labs/l108-reproduction.md','budget':{'limit_usd':10,'recorded_resource_estimate_usd':b['resource_estimate_usd'],'overhead':'UNITEMIZED_RESERVE','provider_invoice':'NOT_CHECKED'},'training':'REUSED_L103_NOT_RUN_IN_L108','full_paper':'NOT_ESTABLISHED','historical_identity':'INCOMPARABLE','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED','learner_status':'PENDING_WRITTEN_DEFENSE','commands':['.venv/bin/modal run modal/l108_replay.py --pilot','.venv/bin/modal run --detach modal/l108_replay.py','.venv/bin/python labs/_collect_l108.py --download','.venv/bin/python labs/_execute_l108.py','.venv/bin/python labs/_delivery_l108.py']}
p.write_text(json.dumps(ledger,indent=2)+'\n');print('Recorded verified L108 scope')
