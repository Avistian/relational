"""Reconcile completed L126 evidence and fingerprint the authored package."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
names=['check','source_check','recovery_check','audit','execution','delivery']
reports={name:json.loads((P/f'_{name}_l126_results.json').read_text()) for name in names}
assert all(r['status']=='PASS' for r in reports.values())
summary=json.loads((P/'evidence/l126/summary.json').read_text())
assert summary['schema']['total_rows']==reports['audit']['independent_sql_rows']==74063
assert reports['audit']['independent_task_rows']==sum(summary['splits'].values())==8712
assert reports['audit']['independently_scored_predictions']==summary['splits']['test']==760
assert reports['source_check']['task_cases']==82 and reports['source_check']['ap_cases']==200
assert len(reports['audit']['mutations'])==6
assert summary['historical_beta_full_contract']=='NOT_RUN'
assert json.loads((P/'_recovery_l126_results.json').read_text())['status']=='BLOCKED_DATA'
data=json.loads((P/'_data_l126.json').read_text())
for name,item in data['files'].items():
    assert hashlib.sha256((P/'evidence/l126'/name).read_bytes()).hexdigest()==item['sha256']==item['registry_sha256']
budget=json.loads((P/'_budget_l126.json').read_text());assert budget['paid_spend_usd']==0 and budget['paid_workers']==0
manifest=json.loads((R/'lessons/manifest.json').read_text());entry=next(x for x in manifest['lessons'] if x['id']==126)
assert entry['labPath']=='labs/0126-relbench-beta.ipynb' and entry['published']
paths=list(P.glob('_*l126*.py'))+[P/'relkit/beta_l126.py',R/'lessons/content/0126-relbench-beta.md',R/'lessons/0126-relbench-beta.html',R/'reference/relbench-beta.html',R/'assets/average-precision-viz.js',R/'assets/l126-lesson.js',P/'0126-relbench-beta.ipynb',P/'solutions/0126-relbench-beta.ipynb',P/'l126-reproduction.md']
r={'status':'PASS','lesson':126,'scope':'API/evaluation contract; no new model','checks':names,'beta_source_cases':82,'ap_cases':200,'full_modern_db_rows':74063,'full_modern_task_rows':8712,'independently_scored_predictions':760,'mutations_rejected':6,'historical_full_contract':'NOT_RUN','blocker':'Checksum-matching beta archives not recovered; seven endpoint probes404','numerical_beta_target':'NOT_APPLICABLE','historical_identity':'NOT_ESTABLISHED','paid_spend_usd':0,'learner_status':'PENDING_WRITTEN_DEFENSE','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED','files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(P/'_verify_l126_results.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='files'})
