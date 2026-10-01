"""Independent metric oracles and adversarial integrity checks."""
import copy,hashlib,json,tempfile,statistics
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,mean_absolute_error
from _replay_l158 import replay
from _replay_l154 import verify_inputs,aligned_score
from _check_l158 import check,rejects
P=Path(__file__).resolve().parent;E=P/'evidence/l158';m=json.loads((E/'input-manifest.json').read_text());r=replay(P,m)
check();n=0;errors=[]
for name in m['files']:
 if not name.endswith('/predictions.npz'):continue
 z=np.load(P/name)
 for split in ['val','test']:
  if split+'_pred' not in z:continue
  pred=z[split+'_pred']
  if '/l153/' in name:
   import gzip
   truth=json.loads(gzip.decompress((P/'evidence/l153/prepared/val-truth.json.gz').read_bytes()))
   lookup={(int(x['entity']),int(x['time'])):set(x['positives']) for x in truth}
   ap=[]
   for entity,time,ranking in zip(z['val_entity'],z['val_time'],pred):
    positives=lookup[(int(entity),int(time))]
    hits=np.isin(ranking,list(positives)).astype(float)
    ap.append(float((np.cumsum(hits)/np.arange(1,11)*hits).sum()/min(len(positives),10)))
   expected=float(np.mean(ap));actual=r['portfolio']['pilot']['mean']
  else:
   y=z[split+'_target'];metric=roc_auc_score if '/l151/' in name else mean_absolute_error
   expected=float(metric(y,pred));result=json.loads((P/name).with_name('result.json').read_text())
   actual=result['scores'][split];actual=actual['roc_auc'] if isinstance(actual,dict) else actual
  errors.append(abs(expected-actual));n+=len(pred)
assert n==r['prediction_rows_rescored'] and max(errors)<1e-10
# Recompute conditional driver bootstrap with a loop instead of the report's matrix reduction.
d=[];entities=None
for seed in range(5):
 a=np.load(P/f'evidence/l155/fe/paper/seed-{seed}/predictions.npz');b=np.load(P/f'evidence/l155/paper/seed-{seed}/predictions.npz')
 g={(int(e),int(t)):float(p) for e,t,p in zip(b['test_entity'],b['test_time'],b['test_pred'])}
 d.append([abs(float(y)-float(p))-abs(float(y)-g[(int(e),int(t))]) for e,t,y,p in zip(a['test_entity'],a['test_time'],a['test_target'],a['test_pred'])]);entities=a['test_entity']
loss=np.mean(d,axis=0);ids=np.unique(entities);rng=np.random.default_rng(155);boot=[]
for _ in range(2000):
 sampled=rng.integers(0,len(ids),size=len(ids));values=np.concatenate([loss[entities==ids[i]] for i in sampled]);boot.append(float(np.mean(values)))
np.testing.assert_allclose(np.quantile(boot,[.025,.975]),r['matched']['test_benefit_driver_bootstrap_95'],atol=1e-12,rtol=0)
rejects(aligned_score,[(1,1),(1,1)],[1,2],[(1,1),(1,2)],[1,2],'MAE')
rejects(aligned_score,[(1,1)],[1],[(1,2)],[1],'MAE')
bad=copy.deepcopy(m);bad['files'][next(iter(bad['files']))]='0'*64
rejects(replay,P,bad)
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/'x';p.write_text('original');mini={'files':{'x':hashlib.sha256(p.read_bytes()).hexdigest()}}
 verify_inputs(tmp,mini);p.write_text('corrupt');rejects(verify_inputs,tmp,mini);p.unlink();rejects(verify_inputs,tmp,mini)
result=dict(status='PASS',independent_rows=n,max_metric_error=max(errors),bootstrap_oracle='LOOP_MATCH',
            corrupt_missing_wrong_hash_and_key_inputs='REJECTED',learner_contracts='PASS',additional_cloud_spend_usd=0)
(P/'_verify_l158_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
