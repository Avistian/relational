"""Final scientific, source, budget and delivery evidence gates."""
from pathlib import Path
import hashlib,json,ast
import nbformat
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l152'
s=json.loads((E/'summary.json').read_text());b=json.loads((P/'_budget_l152.json').read_text())
assert s['status']=='PASS' and s['predictions']==6295 and s['query_occurrences']==403895
assert len(s['records'])==5 and all(r['epochs']==10 and r['complete'] for r in s['records'])
for name,digest in s['hashes'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
for name,digest in b['source_hashes'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
task_sources=json.loads((P/'_task_sources_l152.json').read_text())['sources']
for name,v in task_sources.items():assert hashlib.sha256((P/'sources/l152'/name).read_bytes()).hexdigest()==v['sha256']
reports={}
for name in ['_checkpoint_l152_results.json','_task_audit_l152_results.json','_audit_l152_results.json','_source_check_l152_results.json','_execution_l152_results.json','_notebook_l152_results.json','_delivery_l152_results.json']:
 r=json.loads((P/name).read_text());assert r['status']=='PASS',name;reports[name]=r
code='\n\n'.join(c.source for c in nbformat.read(P/'solutions/0152-regression-portfolio.ipynb',4).cells if c.cell_type=='code');digest=hashlib.sha256(code.encode()).hexdigest()
assert digest==reports['_execution_l152_results.json']['executed_code_sha256']==reports['_notebook_l152_results.json']['code_sha256']
reserved=sum(x['upper_usd'] for x in b['reservations']);assert reserved+b['overhead_reserve_usd']<=10
r=dict(status='PASS',selected_experiment='COMPLETE',test=s['metrics']['test'],test_score=s['metrics']['paper_score'],primary_predictions=6295,notebook_additional_predictions=6295,reservation_plus_overhead_usd=reserved+b['overhead_reserve_usd'],worker_body_estimate_usd=s['worker_body_usd']+reports['_notebook_l152_results.json']['cost']['worker_body_usd'],invoice='NOT_ITEMIZED',source_hashes='PASS',delivery='PASS',historical_identity='NOT_ESTABLISHED',whole_paper='NOT_RUN',fresh_fe_comparison='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE',live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_verify_l152_results.json').write_text(json.dumps(r,indent=2));print(r)
