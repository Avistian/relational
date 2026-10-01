"""Final evidence integrity, source/budget accounting and delivery summary."""
import hashlib,json,ast
from pathlib import Path
import numpy as np
from _replay_l156 import replay
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l156'
m=json.loads((E/'input-manifest.json').read_text());r=replay(E,m);assert r==json.loads((E/'report.json').read_text())
b=json.loads((P/'_budget_l156.json').read_text())
for name,digest in b['source_hashes'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
body=0
for lane in ['paper','fit_horizon']:
 for seed in range(5):
  p=E/lane/f'seed-{seed}';done=json.loads((p/'completed.json').read_text())
  for name,digest in done['source_hashes'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,(lane,seed,name)
  result=json.loads((p/'result.json').read_text());weight=P/f'results/l156/{lane}/seed-{seed}/selected.pt'
  assert hashlib.sha256(weight.read_bytes()).hexdigest()==result['checkpoint_sha256']
  body+=json.loads((p/'cost.json').read_text())['worker_body_usd']
# The three live functions must reject implementations that erase evidence boundaries.
from _check_l156 import check_observations,check_labels,check_verdict
from relkit.temporal_audit_l156 import audit_observations,audit_label_windows,audit_verdict
check_observations(audit_observations);check_labels(audit_label_windows);check_verdict(audit_verdict)
mutants=[]
def must_fail(check,bad):
 try:check(bad)
 except (AssertionError,KeyError,ValueError):return
 raise AssertionError('Mutant survived')
for name,check,bad in [('always-pass',check_observations,lambda _:dict(status='PASS',counts={'PASS':3,'FAIL':0,'NOT_ESTABLISHED':0})),('wrong-window',check_labels,lambda q,e,h:audit_label_windows(q,e,h+1)),('false-signoff',check_verdict,lambda c,r:dict(status='PASS'))]:must_fail(check,bad);mutants.append(name)
checks={}
for file in ['_execution_l156_results.json','_delivery_l156_results.json','_audit_l156_results.json','_sources_l156.json','_check_l156_results.json']:
 data=json.loads((P/file).read_text());assert data['status']=='PASS';checks[file]='PASS'
full=json.loads((E/'notebook/execution.json').read_text());assert full['status']=='PASS' and full['fits']==10 and full['predictions']==12590
assert full['processor_invariance']=='PASS'
assert full['code_sha256']==json.loads((P/'_execution_l156_results.json').read_text())['executed_code_sha256']
# Rescore collected notebook predictions and verify its ten saved checkpoints too.
count=0
for row in full['records']:
 lane=row['lane'];seed=row['seed'];base=E/f'notebook/l156-full/{lane}/seed-{seed}'
 saved=json.loads((base/'result.json').read_text());a=np.load(base/'predictions.npz')
 assert saved['selected_epoch']==min(saved['trace'],key=lambda x:x['val_mae'])['epoch']
 assert len(saved['trace'])==10 and all(x['train_queries']==7453 for x in saved['trace'])
 weights=P/f'results/l156/notebook/l156-full/{lane}/seed-{seed}/selected.pt'
 assert hashlib.sha256(weights.read_bytes()).hexdigest()==saved['checkpoint_sha256']
 for split in ['val','test']:
  q=np.load(E/f'{split}-labels.npz');truth={(int(e),int(t)*10**9):float(y) for e,t,y in zip(q['entity'],q['time'],q['target'])}
  keys=list(zip(a[split+'_entity'].tolist(),a[split+'_time'].tolist()))
  assert len(set(keys))==len(keys) and set(keys)==set(truth)
  target=np.array([truth[k] for k in keys]);np.testing.assert_allclose(target,a[split+'_target'],rtol=0,atol=1e-12)
  score=float(np.abs(a[split+'_pred']-target).mean());assert abs(score-row['scores'][split])<1e-12;count+=len(keys)
assert count==12590
import nbformat
executed=nbformat.read(P/'results/l156/notebook/executed.ipynb',4)
code='\n\n'.join(c.source.replace('RUN_FULL_REPRODUCTION=True','RUN_FULL_REPRODUCTION=False') if '# @colab-bootstrap' in c.source else c.source for c in executed.cells if c.cell_type=='code')
assert hashlib.sha256(code.encode()).hexdigest()==full['code_sha256']
body+=json.loads((E/'notebook/cost.json').read_text())['worker_body_usd']
reserved=sum(x['upper_usd'] for x in b['reservations'])+b['overhead_reserve_usd'];assert reserved<=10 and sum(x['workers'] for x in b['reservations'])<=12
result=dict(status='PASS',primary_fits=10,notebook_validation_fits=10,notebook_fits_in_primary_means=False,labels=sum(r['label_counts'].values()),sql_values=r['sql_values'],primary_predictions=r['predictions'],notebook_predictions=full['predictions'],budget_reserved_plus_overhead_usd=reserved,worker_body_estimate_usd=body,invoice='NOT_ITEMIZED',rejected_learner_mutants=mutants,processor_future_perturbation='PASS',checks=checks,policy_statuses={k:v['strict_policy_verdict']['status'] for k,v in r['lanes'].items()},source_checkpoint_hashes='PASS',live_colab='NOT_CHECKED',deployment='NOT_CHECKED',learner='PENDING_WRITTEN_DEFENSE')
(P/'_verify_l156_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
