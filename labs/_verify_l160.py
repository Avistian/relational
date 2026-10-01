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
# The inherited oracle is executed afresh here; new exam checks follow.
from _replay_l160 import replay160
from _check_l160 import check160
from relkit.exam_l160 import aligned_losses,observed_effort,exit_gates
import platform,sklearn,time
start=time.perf_counter()
m160=json.loads((P/'evidence/l160/input-manifest.json').read_text())
r160=replay160(P,m160)
assert r160==json.loads((P/'evidence/l160/report.json').read_text())
check160()
# Mutant implementations must fail even if they return structurally plausible results.
def rowwise(tk,y,fk,fp,rk,rp):
    return aligned_losses(tk,y,tk,fp,tk,rp)
def invent_effort(records):
    return dict(status='OBSERVED',ratio_fe_over_rdl=1,minutes={'FE':0,'RDL':0})
def waive_gates(entries,failures,review=None):
    output=exit_gates(entries,failures,review);output['exit']='PASS';return output
mutants=[]
for kwargs in [dict(align=rowwise),dict(effort=invent_effort),dict(gates=waive_gates)]:
    try:check160(**kwargs)
    except (AssertionError,ValueError):mutants.append(next(iter(kwargs)))
    else:raise AssertionError('Learner mutant survived')
# Injected functions must actually be called by the full-data adapter.
calls={'align':0,'effort':0,'gates':0}
def tracked(name,fn):
    def call(*args):calls[name]+=1;return fn(*args)
    return call
live=replay160(P,m160,align=tracked('align',aligned_losses),effort=tracked('effort',observed_effort),gates=tracked('gates',exit_gates))
assert live==r160 and calls==dict(align=10,effort=1,gates=1)
bad160=copy.deepcopy(m160);bad160['files'][next(iter(bad160['files']))]='0'*64
rejects(replay160,P,bad160)
assert r160['assessment']['counts']==dict(declared_tasks=3,completed_tasks=2,matched_fe_tasks=1,observed_effort_tasks=0,temporal_pass_tasks=0)
assert r160['assessment']['exit']=='INCOMPLETE' and r160['assessment']['written_defense']=='PENDING_WRITTEN_DEFENSE'
result.update(exam_contracts='PASS',learner_mutants_rejected=mutants,real_adapter_calls=calls,
              frozen_exam_inputs=len(m160['files']),year4_exit='INCOMPLETE',learner='PENDING_WRITTEN_DEFENSE',
              python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,
              source_sha256={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['relkit/exam_l160.py','_replay_l160.py','_check_l160.py','_verify_l160.py']})
(P/'_verify_l160_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
