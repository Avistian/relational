"""Final evidence/source consistency. Never launches paid work."""
import ast,hashlib,json,re,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
required=['_check','_mutation','_source_check','_task_audit','_pilot','_audit','_execution','_delivery']
reports={}
for name in required:
 p=P/(name+'_l124_results.json');r=json.loads(p.read_text());assert r['status']=='PASS',name;reports[name]=sha(p)
s=json.loads((P/'evidence/l124/summary.json').read_text());b=json.loads((P/'_budget_l124.json').read_text());g=json.loads((P/'_task_audit_l124_results.json').read_text())
assert s['status']=='COMPLETE' and s['lesson']==124 and s['evaluated_queries']==6295
assert s['temporal_audit']['audited_query_occurrences']==403895
assert s['worker_resource_usd']==b['recorded_worker_resource_usd']<10
assert b['maximum_worker_usd']+b['overhead_reserve_usd']<=10 and sum(x['workers'] for x in b['reservations'])==6
for relative,digest in b['source_hashes'].items():assert sha(R/relative)==digest,relative
assert g['total_rows']==8712 and g['raw_result_rows']==26080
assert [g['splits'][k]['rows'] for k in ['train','val','test']]==[7453,499,760]
assert [g['splits'][k]['released_only_rows'] for k in ['train','val','test']]==[955,33,42]
assert sha(P/'evidence/l124/task-inputs.json.gz')==g['task_payload_sha256']
for source in json.loads((P/'_sources_l124.json').read_text())['sources'].values():
 name=next(k for k,v in json.loads((P/'_sources_l124.json').read_text())['sources'].items() if v==source)
 assert sha(P/'sources/l124'/name)==source['sha256']
prior={json.loads(p.read_text()).get('run_uuid') for n in [117,119,120,122,123] for p in (P/f'evidence/l{n}').glob('paper/seed-*/completed.json')};seen=set()
for seed in range(5):
 root=P/f'evidence/l124/paper/seed-{seed}';d=json.loads((root/'completed.json').read_text());t=json.loads((root/'temporal-audit.json').read_text());r=json.loads((root/'result.json').read_text())
 assert d['run_uuid'] not in prior|seen;seen.add(d['run_uuid']);assert d['lesson']==124
 assert t['source_sha256']==d['runner_sha256']==sha(P/'_run_l124.py')
 assert t['audit_sha256']==sha(P/'relkit/batch_audit_l123.py')
 assert t['status']=='PASS' and [t['splits'][x]['queries'] for x in ['train','val','test']]==[74530,5489,760]
 assert r['checkpoint_sha256']==sha(P/f'results/l124/paper/seed-{seed}/selected.pt')
manifest=json.loads((R/'lessons/manifest.json').read_text());entry=[x for x in manifest['lessons'] if x['id']==124];assert len(entry)==1 and entry[0]['labPath']=='labs/0124-entity-task-tables.ipynb'
html=(R/'lessons/0124-entity-task-tables.html').read_text()
assert not re.search(r'\[\[[A-Z_]+(?::[a-z]+)?\]\]',html)
for text in ['PENDING_WRITTEN_DEFENSE','3.179653','4.018438','403,895','8,712']:assert text in html,text
assert subprocess.run(['git','check-ignore','-q','labs/solutions/0124-entity-task-tables.ipynb'],cwd=R).returncode==1
paths=[R/'lessons/0124-entity-task-tables.html',R/'lessons/content/0124-entity-task-tables.md',R/'reference/entity-task-tables.html',P/'0124-entity-task-tables.ipynb',P/'solutions/0124-entity-task-tables.ipynb',P/'html/0124-entity-task-tables.html',P/'relkit/tasks_l124.py',P/'relkit/batch_audit_l123.py',P/'l124-reproduction.md',R/'modal/l124_repro.py']
paths+=sorted((P/'evidence/l124').rglob('*'))+list((R/'assets').glob('*task-table*'))+[R/'assets/l124-lesson.js']
paths+=list((P/'figures/l124').glob('*'))+[P/'_sources_l124.json']
hashes={str(p.relative_to(R)):sha(p) for p in paths if p.is_file()}
r=dict(status='PASS',lesson=124,reports=reports,artifact_hashes=hashes,fresh_run_uuids=len(seen),prior_run_uuids_checked=len(prior),selected_experiment='COMPLETE',whole_paper='NOT_ESTABLISHED',availability_history='NOT_OBSERVED',live_colab='NOT_CHECKED',deployment='NOT_CHECKED',learner='PENDING_WRITTEN_DEFENSE',clean_checkout='NOT_CHECKED: existing unrelated unpublished files; copied workspace Pages checked')
(P/'_verify_l124_results.json').write_text(json.dumps(r,indent=2)+'\n')
p=P/'reproductions/execution_evidence.json';ledger=json.loads(p.read_text());ledger['lesson_124']=dict(status='PREPARED_AND_CHECKED',selected_experiment_status='COMPLETE',selected_experiment=s['experiment'],fresh_seeds=5,validation_mae=s['metrics']['val'],test_mae=s['metrics']['test'],task_rows=8712,temporal_queries=403895,task_audit='labs/_task_audit_l124_results.json',worker_resource_usd=s['worker_resource_usd'],billing_total='NOT_ITEMIZED',protocol='labs/l124-reproduction.md',verification='labs/_verify_l124_results.json',whole_paper='NOT_ESTABLISHED',live_colab='NOT_CHECKED',deployment='NOT_CHECKED',learner='PENDING_WRITTEN_DEFENSE');p.write_text(json.dumps(ledger,indent=2)+'\n')
print({k:v for k,v in r.items() if k not in ['reports','artifact_hashes']})
