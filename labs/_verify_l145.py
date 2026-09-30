"""Scientific/evidence checks; a PASS here does not change temporal FAIL."""
import ast,hashlib,itertools,json,math,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error
from relkit.relgt_contracts_l145 import mix_five,audit_tokens,last_validation_min,keyed_mae
from _check_l145 import check_mix,check_audit,check_selection
P=Path(__file__).resolve().parent;E=P/'evidence/l145'
source=json.loads((E/'source.json').read_text())
for name,h in source['sha256'].items():assert hashlib.sha256((P/'sources/l145'/name).read_bytes()).hexdigest()==h
check_mix(mix_five);check_audit(audit_tokens);check_selection(last_validation_min)
# Every short history including ties matches the original <= selection rule.
for n in range(1,6):
 for vals in itertools.product(range(3),repeat=n):
  best=float('inf');idx=None
  for i,v in enumerate(vals):
   if v<=best:best=v;idx=i
  assert last_validation_min(vals)==idx
mutants=[(check_mix,lambda parts,proj:proj(__import__('torch').cat(parts[::-1],-1))),
(check_audit,lambda keys,rows:[(i,j) for i,row in enumerate(rows) for j,t in enumerate(row) if t is not None and t>max(k[1] for k in keys)]),
(check_selection,lambda vals:int(np.argmin(vals)))]
for check,mutant in mutants:
 try:check(mutant)
 except (AssertionError,ValueError):pass
 else:raise AssertionError('Faulty learner implementation accepted')
a=json.loads((E/'prepared/audit.json').read_text());total=0;violations={}
for split,n in [('train',7453),('val',499),('test',760)]:
 z=np.load(E/f'prepared/{split}-audit.npz');keys=list(zip(z['entity'],z['cutoff']));assert len(keys)==n and len(set(keys))==n
 mask=(z['token_times']!=-1)&(z['token_times']>z['cutoff'][:,None]);assert mask.sum()==a['split_audits'][split]['future_token_occurrences'];violations[split]=int(mask.sum());total+=n
assert total==a['independently_rebuilt_labels']==8712
r=json.loads((E/'pilot-1/result.json').read_text());z=np.load(E/'pilot-1/predictions.npz');assert len(z['val_pred'])==256
assert r['history'][0]['queries']==768 and r['test']=='NOT_RUN' and r['real_batch_original_max_error']<1e-5
assert abs(mean_absolute_error(z['val_target'],z['val_pred'])-r['scores']['val'])<1e-9
assert json.loads((E/'cost-decision.json').read_text())['decision']=='STOP'
assert a['temporal_status']=='FAIL'
# Guard rejects failed temporal evidence before considering training or CUDA.
from _full_l145 import full_search
try:full_search(E/'prepared','unused','unused','unused')
except RuntimeError as e:assert 'Temporal audit failed' in str(e)
else:raise AssertionError('Clean reproduction gate accepted leaked context')
# Inlined source model classes must remain source-equivalent except intentional learner delegation.
source_defs={}
for fn in ['codebook.py','encoders.py','local_module.py','model.py']:
 for n in ast.parse((P/'sources/l145'/fn).read_text()).body:
  if isinstance(n,ast.ClassDef):source_defs[n.name]=n
visible={n.name:n for n in ast.parse((P/'relkit/relgt_l145.py').read_text()).body if isinstance(n,ast.ClassDef)}
for name,node in source_defs.items():
 if name!='RelGT':assert ast.dump(node,include_attributes=False)==ast.dump(visible[name],include_attributes=False),name
m=json.loads((E/'mechanism.json').read_text());assert m['status']=='PASS' and m['parameter_gradient_tensors']==149
execution=json.loads((P/'_execution_l145_results.json').read_text());assert execution['status']=='PASS'
pinned=json.loads((E/'notebook-final/execution.json').read_text());assert pinned['status']=='PASS' and pinned['code_sha256']==execution['executed_code_sha256']
budget=json.loads((P/'_budget_l145.json').read_text());reserved=budget['overhead_reserve_usd']+sum(x['upper_usd'] for x in budget['reservations']);assert reserved<=10
costs=list(E.glob('*-cost.json'))+[E/'notebook/cost.json',E/'notebook-final/cost.json'];measured=sum(json.loads(p.read_text())['worker_body_usd'] for p in costs)
report=dict(status='PASS',full_selected_reproduction='INCOMPLETE',full_fits='NOT_RUN',temporal_audit='FAIL',future_token_occurrences=violations,independently_rebuilt_labels=total,independently_scored_pilot_predictions=256,source_class_ast_checks=len(source_defs)-1,synthetic_gradient_tensors=149,real_batch_output_max_error=r['real_batch_original_max_error'],rejected_mutants=3,standalone_notebook='PASS',pinned_notebook='PASS',conservative_reserved_usd=reserved,recorded_worker_body_usd=measured,invoice='NOT_ITEMIZED',historical_identity='NOT_ESTABLISHED',whole_paper='NOT_RUN',live_colab='NOT_CHECKED',deployment='NOT_CHECKED',learner='PENDING_WRITTEN_DEFENSE')
(P/'_verify_l145_results.json').write_text(json.dumps(report,indent=2));print(report)
