"""Final lesson, source, experiment and delivery verification; no paid work."""
import hashlib,json,re,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
reports={}
for name in ['_check','_mutation','_source_check','_pilot','_audit','_trace_audit','_missing_gradient','_execution','_notebook_gpu','_delivery']:
 p=P/(name+'_l131_results.json');report=json.loads(p.read_text());assert report['status']=='PASS',name;reports[name]=sha(p)
s=json.loads((P/'evidence/l131/summary.json').read_text());b=json.loads((P/'_budget_l131.json').read_text())
assert s['status']=='COMPLETE' and s['lesson']==131 and s['evaluated_queries']==6295
assert s['temporal_audit']['audited_query_occurrences']==403895
assert s['worker_resource_usd']==b['recorded_worker_resource_usd']<10
assert b['maximum_worker_usd']+b['overhead_reserve_usd']<=10 and sum(x['workers'] for x in b['reservations'])==10
assert b['resource_usd_including_failed_worker_bound']+b['overhead_reserve_usd']<10
assert b['status']=='COMPLETE_NO_MORE_DISPATCH'
for relative,digest in b['source_hashes'].items():assert sha(R/relative)==digest,relative
sources=json.loads((P/'_sources_l131.json').read_text())
for relative,digest in sources['files'].items():assert sha(R/relative)==digest,relative
assert sha(P/'sources/l131/paper.html')==sources['paper']['sha256']
prior={json.loads(p.read_text()).get('run_uuid') for n in [117,119,120,122,123,124,127,128,130] for p in (P/f'evidence/l{n}').glob('paper/seed-*/completed.json')};seen=set()
for seed in range(5):
 root=P/f'evidence/l131/paper/seed-{seed}';d=json.loads((root/'completed.json').read_text());t=json.loads((root/'temporal-audit.json').read_text());r=json.loads((root/'result.json').read_text())
 assert d['run_uuid'] not in prior|seen;seen.add(d['run_uuid']);assert d['lesson']==131
 assert t['source_sha256']==d['runner_sha256']==sha(P/'_run_l131.py')
 assert json.loads((root/'stack-trace.json').read_text())['source_sha256']==sha(P/'relkit/stack_l131.py')
 assert t['audit_sha256']==sha(P/'relkit/batch_audit_l123.py')
 assert t['status']=='PASS' and [t['splits'][x]['queries'] for x in ['train','val','test']]==[74530,5489,760]
 assert r['checkpoint_sha256']==sha(P/f'results/l131/paper/seed-{seed}/selected.pt')
manifest=json.loads((R/'lessons/manifest.json').read_text());entry=[x for x in manifest['lessons'] if x['id']==131];assert len(entry)==1 and entry[0]['labPath']=='labs/0131-gnn-tabular-stack.ipynb'
html=(R/'lessons/0131-gnn-tabular-stack.html').read_text();assert not re.search(r'\[\[[A-Z_]+(?::[a-z]+)?\]\]',html)
for word in ['PENDING_WRITTEN_DEFENSE',f"{s['metrics']['val']['mean']:.6f}",f"{s['metrics']['test']['mean']:.6f}",'403,895','6,295','nonfinite']:assert word in html,word
assert subprocess.run(['git','check-ignore','-q','labs/solutions/0131-gnn-tabular-stack.ipynb'],cwd=R).returncode==1
paths=[R/'lessons/0131-gnn-tabular-stack.html',R/'lessons/content/0131-gnn-tabular-stack.md',R/'reference/gnn-tabular-stack.html',P/'0131-gnn-tabular-stack.ipynb',P/'solutions/0131-gnn-tabular-stack.ipynb',P/'html/0131-gnn-tabular-stack.html',P/'relkit/stack_l131.py',P/'l131-reproduction.md',R/'modal/l131_repro.py',R/'modal/l131_notebook_check.py',R/'assets/stack-trace-viz.js',R/'assets/l131-lesson.js']
paths+=sorted((P/'evidence/l131').rglob('*'))+sorted((P/'sources/l131').rglob('*'))+list((P/'figures/l131').glob('*'))+[P/'_sources_l131.json']
hashes={str(p.relative_to(R)):sha(p) for p in paths if p.is_file()}
r=dict(status='PASS',lesson=131,reports=reports,artifact_hashes=hashes,fresh_primary_seeds=len(seen),portable_gpu_gate='FIVE_FULL_FRESH_FITS',successful_worker_resource_usd=b['successful_worker_resource_usd'],failed_and_preempted_worker_bound_usd=b['failed_and_preempted_worker_bound_usd'],prior_run_uuids_checked=len(prior),selected_experiment='COMPLETE',whole_paper='NOT_ESTABLISHED',gradient_health='MATCHED_NONFINITE_ENTRIES_DISCLOSED',live_colab='NOT_CHECKED',deployment='NOT_CHECKED',learner='PENDING_WRITTEN_DEFENSE',clean_checkout='NOT_CHECKED: existing unpublished dependencies; copied workspace Pages checked')
(P/'_verify_l131_results.json').write_text(json.dumps(r,indent=2)+'\n')
p=P/'reproductions/execution_evidence.json';ledger=json.loads(p.read_text());ledger['lesson_131']=dict(status='PREPARED_AND_CHECKED',selected_experiment_status='COMPLETE',selected_experiment=s['experiment'],fresh_seeds=5,validation_mae=s['metrics']['val'],test_mae=s['metrics']['test'],temporal_queries=403895,worker_resource_usd=s['worker_resource_usd'],successful_worker_resource_usd=b['successful_worker_resource_usd'],failed_and_preempted_worker_bound_usd=b['failed_and_preempted_worker_bound_usd'],portable_gpu_gate='FIVE_FULL_FRESH_FITS',billing_total='NOT_ITEMIZED',protocol='labs/l131-reproduction.md',verification='labs/_verify_l131_results.json',whole_paper='NOT_ESTABLISHED',gradient_health='MATCHED_NONFINITE_ENTRIES_DISCLOSED',live_colab='NOT_CHECKED',deployment='NOT_CHECKED',learner='PENDING_WRITTEN_DEFENSE');p.write_text(json.dumps(ledger,indent=2)+'\n')
print({k:v for k,v in r.items() if k not in ['reports','artifact_hashes']})
