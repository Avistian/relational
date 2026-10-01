"""Final evidence gates: scientific scope, source identity, standalone cells and budget."""
import ast,hashlib,json
from pathlib import Path
import nbformat
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l153'
s=json.loads((E/'summary.json').read_text());d=json.loads((E/'cost-decision.json').read_text());b=json.loads((P/'_budget_l153.json').read_text())
assert s['status']=='PASS' and s['independently_rescored_rankings']==37003
assert s['full_selected_reproduction']=='INCOMPLETE' and s['paper_comparison']=='NOT_RUN' and not s['records'] and d['decision']=='STOP'
assert s['pilot']['status']=='PARTIAL_TIMING_PILOT' and set(s['pilot']['scores'])=={'val'} and s['pilot']['audit']['train_batches']==32
assert sum(s['labels']['independently_rebuilt_queries'].values())==733741
assert sum(s['pilot']['first_batch_nonfinite_gradients'].values())==384
assert s['pilot']['audit']['temporal_violations']==0
assert s['pilot']['audit']['positive_collisions']==356 and s['pilot']['audit']['negative_comparisons']==8388608
for name,h in b['source_hashes'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
executed=(E/'pilot/executed.py').read_bytes();assert hashlib.sha256(executed).hexdigest()==s['pilot']['executed_sha256']
source=(P/'_run_l153.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='instrument');ns={};exec(ast.get_source_segment(source,node),ns)
assert ns['instrument']((P/'sources/l153/gnn_link.py').read_text(),True).encode()==executed
reports={}
for name in ['_source_check_l153_results.json','_mechanism_l153_results.json','_audit_check_l153_results.json','_execution_l153_results.json','_notebook_l153_results.json','_delivery_l153_results.json']:
 reports[name]=json.loads((P/name).read_text());assert reports[name]['status']=='PASS',name
book=nbformat.read(P/'solutions/0153-recommendation-portfolio.ipynb',4);code='\n\n'.join(c.source for c in book.cells if c.cell_type=='code');digest=hashlib.sha256(code.encode()).hexdigest()
assert digest==reports['_execution_l153_results.json']['executed_code_sha256']==reports['_notebook_l153_results.json']['code_sha256']
# Confirm every visible source string exactly equals the executed dependency bytes.
nodes=ast.parse(code).body
for variable,name in [('RELEASED_GRAPH','graph.py'),('RELEASED_LOADER','loader.py'),('RELEASED_TRAINER','gnn_link.py')]:
 n=next(n for n in nodes if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==variable for t in n.targets));assert ast.literal_eval(n.value)==(P/'sources/l153'/name).read_text()
reserved=sum(x['upper_usd'] for x in b['reservations'])+b['overhead_reserve_usd'];assert reserved<=10
notebook_cost=sum(json.loads(p.read_text())['worker_body_usd'] for p in E.glob('notebook*/cost.json'))
r=dict(status='PASS',delivery='COMPLETE',selected_experiment='INCOMPLETE',cost_decision='STOP',projected_five_run_compute_usd=d['projected_five_run_compute_usd'],saved_validation_rankings=37003,test_predictions='NOT_RUN',labels=733741,positive_collisions=356,shared_negative_comparisons=8388608,nonfinite_gradient_entries=384,source_and_instrumentation='PASS',notebook_cells=20,default_local_and_pinned_notebooks='PASS',full_notebook_training_gate='NOT_RUN',reserved_plus_overhead_usd=reserved,worker_body_estimate_usd=s['worker_body_usd']+notebook_cost,invoice='NOT_ITEMIZED',historical_identity='NOT_ESTABLISHED',feature_arrival_legality='NOT_ESTABLISHED',whole_paper='NOT_RUN',fresh_manual_fe='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE',live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_verify_l153_results.json').write_text(json.dumps(r,indent=2));print(r)
