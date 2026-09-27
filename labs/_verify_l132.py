"""Verify prepared L132 delivery while preserving the incomplete full-run verdict."""
import hashlib,json,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
subprocess.run([sys.executable,str(P/'_check_l132.py')],check=True)
reports={}
for name in ['_mutation','_source_check','_audit_pilots','_sql_audit','_task_audit','_teaching','_execution','_delivery']:
 p=P/(name+'_l132_results.json');v=json.loads(p.read_text());assert v['status']=='PASS',name;reports[name]=sha(p)
b=json.loads((P/'_budget_l132.json').read_text());s=json.loads((P/'evidence/l132/summary.json').read_text())
assert b['status']=='INCOMPLETE_BUDGET_GATE' and s['status']=='INCOMPLETE'
assert len(s['pilots'])==2 and all(x['epochs']==1 for x in s['pilots'])
assert b['projected_total_with_reserved_overhead_usd']>10
for relative,digest in b['source_hashes'].items():assert sha(R/relative)==digest,relative
assert subprocess.run(['git','check-ignore','-q','labs/solutions/0132-identity-aware-message-passing.ipynb'],cwd=R).returncode==1
paths=[R/'lessons/0132-identity-aware-message-passing.html',R/'lessons/content/0132-identity-aware-message-passing.md',R/'reference/identity-aware-message-passing.html',P/'0132-identity-aware-message-passing.ipynb',P/'solutions/0132-identity-aware-message-passing.ipynb',P/'html/0132-identity-aware-message-passing.html',P/'relkit/identity_l132.py',P/'l132-reproduction.md',R/'modal/l132_repro.py',R/'assets/identity-message-viz.js',R/'assets/l132-lesson.js']
paths+=list((P/'evidence/l132').rglob('*'))+list((P/'sources/l132').rglob('*'))+list((P/'figures/l132').glob('*'))+list(P.glob('_*l132.py'))
r=dict(status='PASS',lesson=132,reports=reports,artifact_hashes={str(p.relative_to(R)):sha(p) for p in paths if p.is_file()},selected_experiment='INCOMPLETE_BUDGET_GATE',pilots=2,notebook_code_cells=30,notebook_address_space_limit_mib=4096,whole_paper='NOT_ESTABLISHED',live_colab='NOT_CHECKED',deployment='NOT_CHECKED',learner='PENDING_WRITTEN_DEFENSE',clean_checkout='NOT_CHECKED: existing unpublished dependencies; copied workspace Pages checked')
(P/'_verify_l132_results.json').write_text(json.dumps(r,indent=2)+'\n')
p=P/'reproductions/execution_evidence.json';ledger=json.loads(p.read_text());ledger['lesson_132']=dict(status='PREPARED_AND_CHECKED',selected_experiment_status='INCOMPLETE',selected_experiment='RelBench v1 Table8 condition-sponsor-run; two one-epoch pilots only',pilots=s['pilots'],full_fit_count=0,projected_total_usd=b['projected_total_with_reserved_overhead_usd'],recorded_worker_resource_usd=b['recorded_worker_resource_usd'],billing='NOT_ITEMIZED',protocol='labs/l132-reproduction.md',verification='labs/_verify_l132_results.json',whole_paper='NOT_ESTABLISHED',live_colab='NOT_CHECKED',deployment='NOT_CHECKED',learner='PENDING_WRITTEN_DEFENSE');p.write_text(json.dumps(ledger,indent=2)+'\n')
print({k:v for k,v in r.items() if k not in ['reports','artifact_hashes']})
