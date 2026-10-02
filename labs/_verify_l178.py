"""Independent raw SQL, pairwise metrics and adversarial learner verification."""
import copy,hashlib,json,math,tempfile
from pathlib import Path
import duckdb,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from _audit_l178 import audit178
from _check_l178 import checks
from relkit.comparison_l178 import keyed_auc,comparison_gate,select_validation
P=Path(__file__).resolve().parent;E=P/'evidence/l178';pins=json.loads((E/'audit-input-manifest.json').read_text());report=json.loads((E/'report.json').read_text())
assert audit178(P,pins,keyed_auc,comparison_gate,select_validation)==report
raw=pd.read_parquet(P/'evidence/l171/db/results.parquet');raw=raw[['driverId','date','statusId']];raw['date']=raw.date.astype('datetime64[ns]');con=duckdb.connect();con.register('results',raw);labels_checked=0
for split in ['train','validation','test']:
 a=np.load(E/'released-task'/(split+'.npz'));queries=pd.DataFrame({'row_id':np.arange(len(a['date'])),'driverId':a['driverId'],'cutoff':pd.to_datetime(a['date'])});con.register('queries',queries)
 rows=con.execute('''select q.row_id, max(case when r.statusId != 1 then 1 else 0 end) AS target_value, count(*) n from queries q join results r on q.driverId=r.driverId and r.date>q.cutoff and r.date<=q.cutoff+interval '30 days' group by q.row_id order by q.row_id''').fetchdf()
 assert len(rows)==len(queries) and np.array_equal(rows.row_id,np.arange(len(queries)))
 assert np.array_equal(rows.target_value.to_numpy(),1-a['did_not_finish']);labels_checked+=len(rows)
checked=0;maximum=0
for phase in ['pilot-2','full-1']:
 folder=P/'evidence/l166'/phase
 for record in json.loads((folder/'receipt.json').read_text())['records']:
  a=np.load(folder/(record['arm']+'-'+str(record['seed'])+'.npz'));y=a['label'];p=a['probability'].astype(float);pos=p[y==1,None];neg=p[y==0][None,:]
  pair=float(((pos>neg)+.5*(pos==neg)).mean());sk=float(roc_auc_score(y,p));ours=keyed_auc(a['keys'],y,a['keys'],p)
  assert abs(pair-sk)<1e-12 and abs(pair-ours)<1e-12;maximum=max(maximum,abs(pair-ours));checked+=len(y)
rng=np.random.default_rng(178)
for trial in range(100):
 n=int(rng.integers(4,60));y=rng.integers(0,2,n);y[:2]=[0,1];p=rng.integers(0,11,n)/10;k=np.column_stack([np.arange(n)//3,np.arange(n)%3]);perm=rng.permutation(n)
 assert abs(keyed_auc(k,y,k[perm],p[perm])-roc_auc_score(y,p))<1e-12
 for flip in [False,True]:
  assert abs(keyed_auc(k,1-y,k,1-p)-keyed_auc(k,y,k,p))<1e-12
for trial in range(100):
 rows=[dict(config=c,seed=s,epoch=e,validation_auc=float(rng.random()),test_auc=float(rng.random())) for c in ['a','b'] for s in [0,1,2] for e in range(1,11)]
 best={}
 for c in ['a','b']:
  for s in [0,1,2]:best[c,s]=max((r for r in rows if r['config']==c and r['seed']==s),key=lambda r:(r['validation_auc'],-r['epoch']))
 selected=max(['a','b'],key=lambda c:math.fsum(best[c,s]['validation_auc'] for s in [0,1,2])/3)
 result=select_validation(rows,['a','b'],[0,1,2],10);assert result['config']==selected
 assert result['epochs']=={str(s):best[selected,s]['epoch'] for s in [0,1,2]}
wrong=[(lambda *a:.5,comparison_gate,select_validation),(keyed_auc,lambda *a:dict(status='READY_FOR_PILOT',reasons=[]),select_validation),(keyed_auc,comparison_gate,lambda *a:dict(config='b',epochs={'0':1,'1':1,'2':1},mean_validation_auc=.99))]
for fns in wrong:
 try:checks(*fns)
 except (AssertionError,ValueError):pass
 else:raise AssertionError('Wrong learner implementation accepted')
# Independent gate witness from the actual raw columns.
f=pd.read_parquet(E/'gradient-input.parquet');g=report['fresh']['gradient'];assert f[g['numerical_columns']].isna().any(axis=0).sum()*128==256
assert g['nonfinite_gradient_elements']==256 and report['fresh']['model_runs']==0
assert report['fresh']['status']=='INCOMPLETE_TRAINING_HEALTH_GATE'
result=dict(status='PASS',raw_sql_labels_checked=labels_checked,predictions_checked=checked,independent_metrics=['all-pairs AUROC','sklearn'],max_metric_delta=maximum,randomized_keyed_scores=100,randomized_tuning_grids=100,wrong_learner_functions_rejected=3,gradient_missing_columns=2,gradient_nonfinite_elements=256,fresh_comparison='INCOMPLETE_TRAINING_HEALTH_GATE',cloud_usd=0)
(P/'_verify_l178_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
