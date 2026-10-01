"""Final measured evidence, independent checks and aggregate budget accounting."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error
P=Path(__file__).resolve().parent;D=P/'releases/l157-f1-audit';E=P/'evidence/l157'
sys.path.insert(0,str(D));from cli import source_check
source_check();r=json.loads((E/'report.json').read_text());assert r['contribution']['fits']==10
b=json.loads((D/'budget.json').read_text());reserved=sum(x['upper_usd'] for x in b['reservations'])+b['overhead_reserve_usd'];assert reserved<=10
body=sum(json.loads((E/lane/f'seed-{seed}/cost.json').read_text())['worker_body_usd'] for lane in ['paper','fit_horizon'] for seed in range(5))
checks={}
for name in ['_check_l157_results.json','_package_l157_results.json','_execution_l157_results.json','_delivery_l157_results.json','_audit_l157_results.json','_checkout_l157_results.json']:
 data=json.loads((P/name).read_text());assert data['status']=='PASS',name;checks[name]='PASS'
archive=hashlib.sha256((P/'releases/l157-f1-audit.zip').read_bytes()).hexdigest();assert archive==json.loads((P/'_package_l157_results.json').read_text())['archive_sha256']
full=json.loads((E/'notebook/execution.json').read_text());assert full['status']=='PASS' and full['fits']==10 and not full['primary_mean_inclusion']
assert full['original_code_sha256']==json.loads((P/'_execution_l157_results.json').read_text())['executed_code_sha256']
body+=json.loads((E/'notebook/cost.json').read_text())['worker_body_usd']
# Independently score collected full-notebook outputs, which never enter primary means.
count=0;weights=0
for lane in ['paper','fit_horizon']:
 for seed in range(5):
  base=E/f'notebook/fresh/{lane}/seed-{seed}';saved=json.loads((base/'result.json').read_text());a=np.load(base/'predictions.npz')
  assert saved['selected_epoch']==min(saved['trace'],key=lambda x:x['val_mae'])['epoch']
  assert len(saved['trace'])==10 and all(x['train_queries']==7453 for x in saved['trace'])
  for split in ['val','test']:
   q=np.load(E/f'{split}-labels.npz');truth={(int(e),int(t)*10**9):float(y) for e,t,y in zip(q['entity'],q['time'],q['target'])}
   keys=list(zip(a[split+'_entity'].tolist(),a[split+'_time'].tolist()));assert len(set(keys))==len(keys) and set(keys)==set(truth)
   expected=np.array([truth[k] for k in keys]);np.testing.assert_allclose(expected,a[split+'_target'],rtol=0,atol=1e-12)
   assert abs(mean_absolute_error(expected,a[split+'_pred'])-saved['scores'][split])<1e-12;count+=len(keys)
  check=json.loads((base/'collected-weight.json').read_text());assert check['sha256']==saved['checkpoint_sha256'];weights+=1
assert count==12590 and weights==10
out=dict(status='PASS',experiment=r['experiment'],primary_fits=10,primary_predictions=12590,label_queries=8712,sql_values=r['sql_values'],notebook_full_fits=10,notebook_predictions=count,notebook_primary_mean_inclusion=False,checkpoints_verified=20,checks=checks,reserved_resources_plus_overhead_usd=reserved,measured_worker_body_estimate_usd=body,invoice='NOT_ITEMIZED',initial_notebook_image_build='Local image-order failure followed by remote import failure before notebook cells; app stopped and full failed allocation charged; corrected retry reserved',archive_sha256=archive,publication='PENDING_PUBLICATION',upstream_submission='NOT_SENT',historical_availability='NOT_ESTABLISHED',whole_paper='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE',live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_verify_l157_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
