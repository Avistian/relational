"""Final evidence consistency, freshness and artifact manifest; no new paid runs."""
import hashlib,json,subprocess,re
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent
required=['_check','_mutation','_source_check','_graph_audit','_pilot','_audit','_execution','_delivery']
reports={}
for name in required:
 path=P/(name+'_l122_results.json');r=json.loads(path.read_text());assert r['status']=='PASS',name;reports[name]=hashlib.sha256(path.read_bytes()).hexdigest()
s=json.loads((P/'evidence/l122/summary.json').read_text());b=json.loads((P/'_budget_l122.json').read_text());g=json.loads((P/'_graph_audit_l122_results.json').read_text())
assert s['lesson']==122 and s['status']=='COMPLETE' and s['evaluated_queries']==6295
assert s['worker_resource_usd']==b['recorded_worker_resource_usd']<10
assert b['maximum_worker_usd']+b['overhead_reserve_usd']<=10 and sum(x['workers'] for x in b['reservations'])==6
for rel,digest in b['source_hashes'].items():assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==digest
assert g['total_rows']==74063 and g['directed_edges']==338842
assert hashlib.sha256((P/'relkit/reg_l122.py').read_bytes()).hexdigest()==g['source_sha256']
assert hashlib.sha256((P/'evidence/l122/reg-topology.npz').read_bytes()).hexdigest()==g['topology_sha256']
prior=set()
for lesson in [117,119,120]:
 for file in (P/f'evidence/l{lesson}').glob('paper/seed-*/completed.json'):
  prior.add(json.loads(file.read_text()).get('run_uuid'))
new=set()
for seed in range(5):
 d=json.loads((P/f'evidence/l122/paper/seed-{seed}/completed.json').read_text());assert d['run_uuid'] not in prior|new;new.add(d['run_uuid'])
 a=json.loads((P/f'evidence/l122/paper/seed-{seed}/audit.json').read_text());assert a['rows']==g['rows']
 for relation in g['forward_relations']:
  src,rel,dst=relation['type'];assert a['edges']['|'.join((src,rel,dst))]==relation['edges']==a['edges']['|'.join((dst,'rev_'+rel,src))]
manifest=json.loads((R/'lessons/manifest.json').read_text());entry=[x for x in manifest['lessons'] if x['id']==122];assert len(entry)==1
html=(R/'lessons/0122-reg-construction.html').read_text();assert not re.search(r'\[\[[A-Z_]+(?::[a-z]+)?\]\]',html)
for text in ['PENDING_WRITTEN_DEFENSE','3.188699','4.055114','74063','338842']:assert text in html,text
solution='labs/solutions/0122-reg-construction.ipynb'
assert subprocess.run(['git','check-ignore','-q',solution],cwd=R).returncode==1,'Solution ignored'
files=[R/'lessons/0122-reg-construction.html',R/'reference/reg-construction.html',P/'0122-reg-construction.ipynb',R/solution,P/'html/0122-reg-construction.html',P/'relkit/reg_l122.py',P/'l122-reproduction.md',R/'modal/l122_repro.py']
files+=sorted((P/'evidence/l122').rglob('*'))
hashes={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()}
r={'status':'PASS','lesson':122,'reports':reports,'artifact_hashes':hashes,'fresh_run_uuids':len(new),'prior_run_uuids_checked':len(prior),'graph_counts_match_every_training_run':True,'selected_experiment':'COMPLETE','whole_paper':'NOT_ESTABLISHED','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED','learner':'PENDING_WRITTEN_DEFENSE'}
(P/'_verify_l122_results.json').write_text(json.dumps(r,indent=2)+'\n')
f=P/'reproductions/execution_evidence.json';ledger=json.loads(f.read_text());ledger['lesson_122']={'status':'PREPARED_AND_CHECKED','selected_experiment':s['experiment'],'selected_experiment_status':'COMPLETE','graph_audit':'labs/_graph_audit_l122_results.json','graph_rows':74063,'directed_edges':338842,'fresh_seeds':5,'validation_mae':s['metrics']['val'],'test_mae':s['metrics']['test'],'worker_resource_usd':s['worker_resource_usd'],'billing_total':'NOT_ITEMIZED','verification':'labs/_verify_l122_results.json','protocol':'labs/l122-reproduction.md','whole_paper':'NOT_ESTABLISHED','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED','learner':'PENDING_WRITTEN_DEFENSE'};f.write_text(json.dumps(ledger,indent=2)+'\n')
print({k:v for k,v in r.items() if k not in ['reports','artifact_hashes']})
