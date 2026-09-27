"""Verify final artifacts, fresh execution identities and evidence boundaries; no paid work."""
import hashlib,json,re,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
reports={}
for name in ['_check','_mutation','_source_check','_pilot','_audit','_execution','_delivery']:
 p=P/(name+'_l127_results.json');r=json.loads(p.read_text());assert r['status']=='PASS',name;reports[name]=sha(p)
s=json.loads((P/'evidence/l127/summary.json').read_text());b=json.loads((P/'_budget_l127.json').read_text())
assert s['status']=='COMPLETE' and s['lesson']==127 and s['evaluated_queries']==6295
assert s['temporal_audit']['audited_query_occurrences']==403895
assert s['worker_resource_usd']==b['recorded_worker_resource_usd']<10
assert b['maximum_worker_usd']+b['overhead_reserve_usd']<=10 and sum(x['workers'] for x in b['reservations'])==6
for relative,digest in b['source_hashes'].items():assert sha(R/relative)==digest,relative
sources=json.loads((P/'_sources_l127.json').read_text())
for relative,digest in sources['files'].items():assert sha(R/relative)==digest,relative
assert sha(P/'sources/l127/paper.html')==sources['paper']['sha256']
prior={json.loads(p.read_text()).get('run_uuid') for n in [117,119,120,122,123,124] for p in (P/f'evidence/l{n}').glob('paper/seed-*/completed.json')};seen=set()
for seed in range(5):
 root=P/f'evidence/l127/paper/seed-{seed}';d=json.loads((root/'completed.json').read_text());t=json.loads((root/'temporal-audit.json').read_text());r=json.loads((root/'result.json').read_text())
 assert d['run_uuid'] not in prior|seen;seen.add(d['run_uuid']);assert d['lesson']==127
 assert t['source_sha256']==d['runner_sha256']==sha(P/'_run_l127.py')
 assert t['audit_sha256']==sha(P/'relkit/batch_audit_l123.py')
 assert t['status']=='PASS' and [t['splits'][x]['queries'] for x in ['train','val','test']]==[74530,5489,760]
 assert r['checkpoint_sha256']==sha(P/f'results/l127/paper/seed-{seed}/selected.pt')
manifest=json.loads((R/'lessons/manifest.json').read_text());entry=[x for x in manifest['lessons'] if x['id']==127];assert len(entry)==1 and entry[0]['labPath']=='labs/0127-relbench-v1.ipynb'
html=(R/'lessons/0127-relbench-v1.html').read_text()
assert not re.search(r'\[\[[A-Z_]+(?::[a-z]+)?\]\]',html)
for text in ['PENDING_WRITTEN_DEFENSE','3.178174','3.967165','403,895','6,295']:assert text in html,text
assert subprocess.run(['git','check-ignore','-q','labs/solutions/0127-relbench-v1.ipynb'],cwd=R).returncode==1
paths=[R/'lessons/0127-relbench-v1.html',R/'lessons/content/0127-relbench-v1.md',R/'reference/relbench-v1.html',P/'0127-relbench-v1.ipynb',P/'solutions/0127-relbench-v1.ipynb',P/'html/0127-relbench-v1.html',P/'relkit/benchmark_l127.py',P/'relkit/batch_audit_l123.py',P/'l127-reproduction.md',R/'modal/l127_repro.py']
paths+=sorted((P/'evidence/l127').rglob('*'))+[R/'assets/checkpoint-audit-viz.js',R/'assets/l127-lesson.js']
paths+=list((P/'figures/l127').glob('*'))+[P/'_sources_l127.json']
hashes={str(p.relative_to(R)):sha(p) for p in paths if p.is_file()}
r=dict(status='PASS',lesson=127,reports=reports,artifact_hashes=hashes,fresh_run_uuids=len(seen),prior_run_uuids_checked=len(prior),selected_experiment='COMPLETE',whole_paper='NOT_ESTABLISHED',availability_history='NOT_OBSERVED',live_colab='NOT_CHECKED',deployment='NOT_CHECKED',learner='PENDING_WRITTEN_DEFENSE',clean_checkout='NOT_CHECKED: pre-existing unpublished files; copied workspace Pages checked')
(P/'_verify_l127_results.json').write_text(json.dumps(r,indent=2)+'\n')
p=P/'reproductions/execution_evidence.json';ledger=json.loads(p.read_text());ledger['lesson_127']=dict(status='PREPARED_AND_CHECKED',selected_experiment_status='COMPLETE',selected_experiment=s['experiment'],fresh_seeds=5,validation_mae=s['metrics']['val'],test_mae=s['metrics']['test'],temporal_queries=403895,worker_resource_usd=s['worker_resource_usd'],billing_total='NOT_ITEMIZED',protocol='labs/l127-reproduction.md',verification='labs/_verify_l127_results.json',whole_paper='NOT_ESTABLISHED',live_colab='NOT_CHECKED',deployment='NOT_CHECKED',learner='PENDING_WRITTEN_DEFENSE');p.write_text(json.dumps(ledger,indent=2)+'\n')
print({k:v for k,v in r.items() if k not in ['reports','artifact_hashes']})
