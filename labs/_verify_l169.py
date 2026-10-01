"""Independent metric, aggregation, boundary and corrupt-evidence checks; no inference."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import copy,hashlib,json,math,shutil,signal,tempfile
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from _audit_l169 import audit169
from _check_l169 import check169
from relkit.transfer_l167 import keyed_auc
from relkit.scaling_l169 import sample_context,scaling_curve,scaling_claim
signal.alarm(600);P=Path(__file__).resolve().parent;E=P/'evidence/l169'
assert check169(sample_context,scaling_curve,scaling_claim)=='PASS'
wrong=[(lambda n,k,t:np.arange(k),scaling_curve,scaling_claim),(sample_context,lambda r:dict(levels=[],doublings=[]),scaling_claim),(sample_context,scaling_curve,lambda *a:'SCALING_LAW')]
for funcs in wrong:
 try:check169(*funcs)
 except (AssertionError,ValueError,IndexError):pass
 else:raise AssertionError('Incorrect learner solution accepted')
rng=np.random.default_rng(169);metric_error=0
for n in range(2,102):
 y=np.r_[0,1,rng.integers(0,2,n-2)];scores=rng.integers(0,9,n)/8;keys=np.column_stack([np.arange(n)//2,np.arange(n)%2]);order=rng.permutation(n)
 actual=keyed_auc(keys,y,keys[order],scores[order]);pos=scores[y==1,None];neg=scores[None,y==0]
 oracle=float(np.mean((pos>neg)+.5*(pos==neg)))
 metric_error=max(metric_error,abs(actual-oracle),abs(actual-roc_auc_score(y,scores)));assert metric_error<1e-12
cases=0
for axis in ['context','parameters','pretraining_data','schema_diversity']:
 for controlled in [False,True]:
  for levels in [1,5]:
   for extra in [False,True]:
    got=scaling_claim(axis,controlled,levels,extra)
    expected='CONFOUNDED' if not controlled else 'INSUFFICIENT_LEVELS' if levels==1 else 'EXTRAPOLATION_NOT_ESTABLISHED' if extra else 'CONTEXT_RESPONSE_ONLY' if axis=='context' else 'CONTROLLED_SWEEP_NOT_LAW'
    assert got==expected;cases+=1
pins=json.loads((E/'audit-manifest.json').read_text());report=audit169(P,pins,keyed_auc,sample_context,scaling_curve,scaling_claim)
assert report==json.loads((E/'report.json').read_text())
raw_count=0;run_count=0
for name in pins['files']:
 if not name.endswith('receipt.json') or '/tail-1/' in name:continue
 receipt=json.loads((P/name).read_text())
 for r in receipt['records']:
  filename=r.get('filename',f"{r['arm']}-{r['seed']}.npz")
  path=(P/name).parent/filename
  with np.load(path) as a:
   assert abs(roc_auc_score(a['label'],a['probability'])-r['auc'])<1e-12;raw_count+=len(a['label']);run_count+=1
assert raw_count==229050 and run_count==300
for c in report['curves'].values():
 for level in c['levels']:
  values=level['per_seed'];mean=math.fsum(values)/10;sd=math.sqrt(math.fsum((v-mean)**2 for v in values)/9)
  assert abs(mean-level['mean'])<1e-12 and abs(sd-level['sample_sd'])<1e-12
 for i,d in enumerate(c['doublings']):
  expected=[b-a for a,b in zip(c['levels'][i]['per_seed'],c['levels'][i+1]['per_seed'])]
  np.testing.assert_allclose(expected,d['per_seed'],atol=1e-14);assert abs(math.fsum(expected)/10-d['mean_gain'])<1e-12
# Test randomized aggregation order and non-monotone curves, independently of saved means.
for _ in range(100):
 values=rng.uniform(.3,.9,size=(5,10));rows=[dict(context=k,seed=s,auc=float(values[i,s])) for i,k in enumerate([64,128,256,512,1024]) for s in range(10)];rng.shuffle(rows)
 c=scaling_curve(rows)
 np.testing.assert_allclose([l['mean'] for l in c['levels']],values.mean(axis=1))
 np.testing.assert_allclose([d['mean_gain'] for d in c['doublings']],np.diff(values,axis=0).mean(axis=1))
corruptions=0
with tempfile.TemporaryDirectory(prefix='l169-corrupt-') as tmp:
 root=Path(tmp)
 for name in pins['files']:
  p=root/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/name,p)
 rn='evidence/l169/remaining-complete/receipt.json';rp=root/rn;original_receipt=rp.read_bytes();r0=json.loads(original_receipt)['records'][0];pn='evidence/l169/remaining-complete/'+r0['filename'];pp=root/pn;original_pred=pp.read_bytes()
 for mode in ['missing','duplicate','wrong_support','wrong_keys','wrong_labels','bad_hash']:
  changed=copy.deepcopy(pins);receipt=json.loads(original_receipt)
  if mode=='missing':receipt['records'].pop()
  elif mode=='duplicate':receipt['records'].append(receipt['records'][0])
  elif mode=='bad_hash':pp.write_bytes(b'corrupt')
  else:
   with np.load(pp) as raw:arr={k:raw[k] for k in raw.files}
   if mode=='wrong_support':arr['support_keys'][0]=arr['support_keys'][1]
   elif mode=='wrong_keys':arr['keys'][0]=arr['keys'][1]
   else:arr['label'][0]=1-arr['label'][0]
   np.savez_compressed(pp,**arr);digest=hashlib.sha256(pp.read_bytes()).hexdigest();changed['files'][pn]=digest;receipt['records'][0]['sha256']=digest
  rp.write_text(json.dumps(receipt));changed['files'][rn]=hashlib.sha256(rp.read_bytes()).hexdigest()
  try:audit169(root,changed,keyed_auc,sample_context,scaling_curve,scaling_claim)
  except ValueError:corruptions+=1
  else:raise AssertionError('Corruption accepted: '+mode)
  rp.write_bytes(original_receipt);pp.write_bytes(original_pred)
# Budget guard checks happen against copies; never mutate the live reservation ledger.
from _guard_l166 import reserve
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);(root/'worker.py').write_text('pass\n');digest=hashlib.sha256((root/'worker.py').read_bytes()).hexdigest();b=root/'budget.json'
 base=dict(cap_usd=1,overhead_reserve_usd=.5,reservations=[],source_hashes={'worker.py':digest})
 b.write_text(json.dumps(base));reserve(b,root,'first',10)
 for phase,seconds in [('first',10),('overspend',5000)]:
  try:reserve(b,root,phase,seconds)
  except ValueError:pass
  else:raise AssertionError('Invalid reservation accepted')
 (root/'worker.py').write_text('print(1)\n')
 try:reserve(b,root,'changed',10)
 except ValueError:pass
 else:raise AssertionError('Changed worker accepted')
r=dict(status='PASS',complete_runs=run_count,raw_predictions=raw_count,rank_sklearn_pairwise_cases=100,max_metric_discrepancy=metric_error,
 curve_order_cases=100,claim_cases=cases,rejected_bad_learner_functions=3,rejected_corrupt_packets=corruptions,budget_duplicate_cap_source_checks='PASS',hashed_files=len(pins['files']))
(P/'_verify_l169_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
