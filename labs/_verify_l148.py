"""Scientific evidence completeness, source identity and active contract checks."""
import json,hashlib,subprocess
from pathlib import Path
import numpy as np
from _check_l148 import check_history,check_keyed,check_effect
from relkit.ablation_l148 import history_mask,keyed_mae,interaction_summary
P=Path(__file__).resolve().parent;E=P/'evidence/l148';s=json.loads((E/'summary.json').read_text());b=json.loads((P/'_budget_l148.json').read_text())
for check,fn in [(check_history,history_mask),(check_keyed,keyed_mae),(check_effect,interaction_summary)]:check(fn)
def wrong_owner(c,t,o,u,r,w):return history_mask(c,t,np.zeros_like(o),u,r,w)
def wrong_boundary(c,t,o,u,r,w):return history_mask(c,t,o,u,r,w-1)
def wrong_position(a,y,k,p):return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))
def wrong_interaction(f,e,m,c):return dict(differences=[0,0],mean=0,sample_sd=0)
rejected=[]
for check,fn in [(check_history,wrong_owner),(check_history,wrong_boundary),(check_keyed,wrong_position),(check_effect,wrong_interaction)]:
 try:check(fn)
 except (AssertionError,ValueError):rejected.append(fn.__name__)
assert len(rejected)==4
assert s['verified_predictions']==31475 and len(s['runs'])==25 and len({r['run_id'] for r in s['runs']})==25
for name,digest in s['artifact_hashes'].items():assert hashlib.sha256((E/name).read_bytes()).hexdigest()==digest
upper=3+sum(x['upper_usd'] for x in b['reservations']);assert upper<=10
for name in ['sources.json','label-audit.json','neural.json','neural-pinned.json','real-probe.json']:
 assert json.loads((E/name).read_text())['status']=='PASS'
# The scientific implementation must not drift between primary dispatches.
training=[r for r in b['reservations'] if r['phase'].startswith(tuple(a+'-' for a in ['full','encoder','messages','history','combined']))]
assert len(training)==25
for path in ['labs/_train_l148.py','labs/relkit/ablation_model_l148.py','labs/relkit/ablation_l148.py']:
 expected=hashlib.sha256((P.parent/path).read_bytes()).hexdigest();assert all(r['sources'][path]==expected for r in training)
result=dict(status='PASS',fresh_fits=25,epochs_per_fit=10,verified_predictions=31475,independent_labels=8712,mutation_rejections=rejected,reservation_plus_overhead_usd=upper,baseline_closeness=s['baseline_closeness'],whole_paper='NOT_RUN',historical_identity='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE')
(P/'_verify_l148_results.json').write_text(json.dumps(result,indent=2));print(result)
