"""Source identity, contracts, evidence scope and aggregate budget verification."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l144'
manifest=json.loads((P/'_sources_l144.json').read_text())
for name,digest in manifest['files'].items():assert hashlib.sha256((P/'sources/l144'/name).read_bytes()).hexdigest()==digest,name
visible={n.name:n for n in ast.parse((P/'relkit/contextgnn_l144.py').read_text()).body if isinstance(n,ast.ClassDef)}
classes=0
for path in (P/'sources/l144/contextgnn').rglob('*.py'):
 for n in ast.parse(path.read_text()).body:
  if not isinstance(n,ast.ClassDef) or n.name not in visible:continue
  original=n;current=visible[n.name]
  if n.name=='ContextGNN':
   original.body=[m for m in original.body if not isinstance(m,ast.FunctionDef) or m.name!='construct_logits'];current.body=[m for m in current.body if not isinstance(m,ast.FunctionDef) or m.name!='construct_logits']
  assert ast.dump(original,include_attributes=False)==ast.dump(current,include_attributes=False),n.name;classes+=1
subprocess.run([sys.executable,str(P/'_check_l144.py')],check=True)
budget=json.loads((P/'_budget_l144.json').read_text());reserved=sum(x['upper_usd'] for x in budget['reservations'])+budget['overhead_reserve_usd'];assert reserved<=10
summary=json.loads((E/'summary.json').read_text());decision=json.loads((E/'cost-decision.json').read_text())
if decision['decision']=='STOP':assert not any(x['phase'].startswith('search-') for x in budget['reservations'])
for r in summary['pilots']:
 assert r['status']=='PARTIAL_TIMING_PILOT' and r['scope']['training_batch_limit']==16 and r['scope']['validation_batch_limit']==4
 assert len(r['history'])==1 and r['history'][0]['steps']==16 and r['history'][0]['queries']==4096
 assert r['temporal_audit']['future_violations']==0 and r['source_parity_max_abs']<=1e-5
 assert 'test' not in r['scores']
assert summary['full_selected_reproduction']=='INCOMPLETE'
labels=json.loads((E/'independent-labels.json').read_text());assert labels['independently_rebuilt_queries']=={'train':669310,'val':37003,'test':27428}
mechanism=json.loads((P/'_mechanism_l144_results.json').read_text());assert mechanism['status']=='PASS'
body_cost=sum(json.loads(path.read_text())['worker_body_usd'] for path in E.rglob('*cost.json'))
report={'status':'PASS','unchanged_source_classes_except_tested_fusion':classes,'source_files_verified':len(manifest['files']),'labels_independently_reconstructed':sum(labels['independently_rebuilt_queries'].values()),'pilot_training_queries':sum(r['history'][0]['queries'] for r in summary['pilots']),'pilot_validation_rows':summary['verified_ranking_rows'],'reserved_plus_overhead_usd':reserved,'worker_body_estimate_usd':body_cost,'invoice':'NOT_ITEMIZED','full_selected_reproduction':'INCOMPLETE','whole_paper':'NOT_RUN','historical_identity':'NOT_ESTABLISHED','live_colab':'NOT_CHECKED','learner':'PENDING_WRITTEN_DEFENSE'}
(P/'_verify_l144_results.json').write_text(json.dumps(report,indent=2));print(report)
