"""Independent metric, split, keyed prediction and ridge-equation verification."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json,itertools
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error
from relkit.rows_l163 import serialize_row,masked_mean,choose_alpha,typed_features
from _check_l163 import check163
P=Path(__file__).resolve().parent;E=P/'evidence/l163'
p=json.loads((E/'fixtures.json').read_text());r=json.loads((E/'report.json').read_text());s=json.loads((E/'selection.json').read_text());receipt=json.loads((E/'encoding-receipt.json').read_text());a=dict(np.load(E/'embeddings.npz'));traces=json.loads((E/'token-traces.json').read_text())
assert check163(serialize_row,masked_mean,choose_alpha,typed_features)=='PASS'
for file,key in [('fixtures.json','fixtures_sha256'),('embeddings.npz','embeddings_sha256'),('token-traces.json','token_traces_sha256')]:assert hashlib.sha256((E/file).read_bytes()).hexdigest()==receipt[key]
assert hashlib.sha256((E/'selection.json').read_bytes()).hexdigest()==json.loads((E/'head-receipt.json').read_text())['selection_sha256']
ledger=json.loads((P/'sources/l163/source-ledger.json').read_text())
for src in ledger['sources']:assert hashlib.sha256((P/'sources/l163'/src['file']).read_bytes()).hexdigest()==src['sha256']
for split in p['splits']:
 groups=[set(split[k]) for k in ['train','validation','test']]
 assert [len(g) for g in groups]==[144,48,48]
 assert set.union(*groups)==set(range(240)) and all(not x&y for x,y in itertools.combinations(groups,2))
assert len({row['id'] for row in p['rows']})==240
assert receipt['trainable_parameters']==0
for variant in p['variants']:
 assert a[variant].shape==(240,768) and np.isfinite(a[variant]).all()
 for row,trace in zip(p['rows'],traces[variant]):
  assert row['id']==trace['id'] and serialize_row(row,variant)==trace['text']
  payload=json.loads(trace['text']);assert len(payload)==5 and 'target' not in payload and 'id' not in payload
  assert trace['length']==len(trace['token_ids'])==len(trace['tokens'])<=1024
actual={(v['seed'],v['method'],v['variant'],v['id']):v for v in r['predictions']}
expected={(split['seed'],method,variant,p['rows'][i]['id']) for split in p['splits'] for method in ['typed','bart'] for variant in p['variants'] for i in split['test']}
assert len(actual)==len(r['predictions'])==864 and set(actual)==expected
max_metric=0.;max_ridge=0.;max_validation=0.
rows=p['rows'];target=np.array([row['target'] for row in rows])
for head in s['heads']:
 split=next(v for v in p['splits'] if v['seed']==head['seed']);train=split['train'];valid=split['validation'];test=split['test']
 # Independent typed feature construction: raw numeric columns, then categories.
 raw=np.array([[row['price_usd'],row['weight_kg']] for row in rows]);mu=raw[train].mean(0);sd=raw[train].std(0);sd[sd==0]=1
 blocks=[(raw-mu)/sd]
 for key in ['colour','condition']:
  vocab=sorted(set(rows[i][key] for i in train));blocks.append(np.array([[int(row[key]==v) for v in vocab] for row in rows]))
 typed=np.column_stack(blocks)
 x=typed if head['method']=='typed' else a['baseline'].astype(float)
 mean=x[train].mean(0);scale=x[train].std(0);scale[scale==0]=1
 np.testing.assert_allclose(mean,head['feature_mean'],atol=1e-10)
 np.testing.assert_allclose(scale,head['feature_scale'],atol=1e-10)
 z=(x-mean)/scale;center=z[train].mean(0);yc=target[train]-target[train].mean();xt=z[train]-center
 validation=[];solutions=[]
 for alpha in p['alphas']:
  dual=np.linalg.solve(xt@xt.T+alpha*np.eye(len(train)),yc);solutions.append(dual)
  vp=(z[valid]-center)@xt.T@dual+target[train].mean();validation.append(float(np.mean(abs(vp-target[valid]))))
 max_validation=max(max_validation,float(np.max(np.abs(np.array(validation)-head['validation_mae']))))
 assert p['alphas'][int(np.argmin(validation))]==head['alpha']
 dual=solutions[p['alphas'].index(head['alpha'])]
 for variant in p['variants']:
  v=typed if head['method']=='typed' else a[variant].astype(float)
  oracle=((v[test]-mean)/scale-center)@xt.T@dual+target[train].mean()
  predicted=np.array([actual[head['seed'],head['method'],variant,rows[i]['id']]['prediction'] for i in test])
  max_ridge=max(max_ridge,float(np.max(np.abs(oracle-predicted))))
  metric=float(mean_absolute_error(target[test],predicted))
  reported=next(v['test_mae'] for v in r['results'] if (v['seed'],v['method'],v['variant'])==(head['seed'],head['method'],variant))
  max_metric=max(max_metric,abs(metric-reported))
  for i,estimate in zip(test,predicted):
   record=actual[head['seed'],head['method'],variant,rows[i]['id']]
   assert record['target']==target[i] and abs(record['absolute_error']-abs(estimate-target[i]))<1e-10
assert max_metric<1e-10
assert max_ridge<1e-6,(max_ridge,max_validation)
assert max_validation<1e-6,max_validation
# CHECK must reject each distinct learner failure mode.
bad=[(lambda row,variant='baseline':json.dumps(row),masked_mean,choose_alpha),(serialize_row,lambda h,m:np.asarray(h).mean(1),choose_alpha),(serialize_row,masked_mean,lambda a,v:a[0])]
for serial,pool,choose in bad:
 try:check163(serial,pool,choose,typed_features)
 except (AssertionError,ValueError):pass
 else:raise AssertionError('Incorrect learner implementation accepted')
# Independent direct masked-index averaging across varied padding patterns.
rng=np.random.default_rng(91)
for n in range(1,10):
 h=rng.normal(size=(5,n,7));mask=rng.integers(0,2,size=(5,n));mask[:,0]=1
 expected=np.stack([h[i,mask[i].astype(bool)].mean(0) for i in range(5)])
 np.testing.assert_allclose(masked_mean(h,mask),expected,atol=1e-12)
# Live notebook pooling fixtures are real checkpoint states with exact cache parity.
states=dict(np.load(E/'pool-states.npz'));pool_receipt=json.loads((E/'pool-receipt.json').read_text())
assert hashlib.sha256((E/'pool-states.npz').read_bytes()).hexdigest()==pool_receipt['sha256']
for variant in p['variants']:
 np.testing.assert_array_equal(masked_mean(states[variant+'_hidden'],states[variant+'_mask']).astype(np.float32),a[variant][:8])
# Aggregate scores are independently recomputed from all split scores.
import math
for summary in r['summary']:
 values=[v['test_mae'] for v in r['results'] if (v['method'],v['variant'])==(summary['method'],summary['variant'])]
 mean=math.fsum(values)/3;sd=math.sqrt(math.fsum((x-mean)**2 for x in values)/2)
 assert abs(mean-summary['mean_mae'])<1e-10 and abs(sd-summary['sample_sd'])<1e-10
result=dict(status='PASS',keyed_predictions=864,selected_heads=6,source_hashes='PASS',split_partitions='PASS',rejected_incorrect_implementations=3,independent_pooling_rows=45,max_metric_difference=max_metric,max_independent_ridge_prediction_difference=max_ridge,max_validation_difference=max_validation,fresh_encoding=receipt['status'],historical_reproduction='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')
(P/'_verify_l163_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
