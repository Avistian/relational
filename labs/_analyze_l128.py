"""Independent scalar MAE, query identity, checkpoint and protocol checks."""
import hashlib,json,math,statistics
from pathlib import Path
import numpy as np
from relkit.taxonomy_l128 import binary_auc
contract=dict(epochs=10,seeds=list(range(5)))
def select_checkpoint(trace,epochs):
 assert len(trace)==epochs and [x["epoch"] for x in trace]==list(range(1,epochs+1))
 return max(trace,key=lambda x:x["val_auc"])["epoch"]
def summarize_seeds(records,seeds):
 assert sorted(x["seed"] for x in records)==seeds
 return {s:dict(mean=statistics.mean(x[s] for x in records),sample_sd=statistics.stdev(x[s] for x in records)) for s in ["val","test"]}
learner_records=[]
P=Path(__file__).resolve().parent
rows=[];maximum=0.;total=0;cost=0.;schema=None
uuids=set()
budget=json.loads((P/'_budget_l128.json').read_text())
for seed in range(5):
 root=P/f'evidence/l128/paper/seed-{seed}'
 r=json.loads((root/'result.json').read_text());a=json.loads((root/'audit.json').read_text());done=json.loads((root/'completed.json').read_text())
 assert done['lesson']==128 and done['seed']==seed and done['status']=='COMPLETE'
 assert done['started_utc']>=next(v['utc'] for v in budget['reservations'] if v['mode']=='paper') and done['finished_utc']>done['started_utc']
 assert done['run_uuid'] not in uuids
 uuids.add(done['run_uuid'])
 assert done['runner_sha256']==hashlib.sha256((P/'_run_l128.py').read_bytes()).hexdigest()
 assert r['seed']==seed
 assert r['epochs']==10 and len(r['trace'])==10 and all(v['train_queries']==11411 for v in r['trace'])
 assert r['selected_epoch']==select_checkpoint(r['trace'],contract['epochs'])
 assert r['checkpoint_sha256']==hashlib.sha256((P/f'results/l128/paper/seed-{seed}/selected.pt').read_bytes()).hexdigest()
 assert a['source_sha256']==hashlib.sha256((P/'relkit/rdl_l117.py').read_bytes()).hexdigest()
 assert sum(a['rows'].values())==74063
 current=(a['stypes'],a['rows'],a['edges'],a['packages'],list(a['rows']))
 if schema is None:schema=current
 else:assert schema==current,'Preprocessing/runtime/table ordering differed across runs'
 pred=np.load(root/'predictions.npz');scores={}
 for split,n in [('val',566),('test',702)]:
  y=pred[split+'_target'];p=pred[split+'_pred'];assert len(y)==n and np.isfinite(p).all()
  metric=binary_auc(y,p)
  assert abs(metric-r['scores'][split])<1e-12
  assert np.all(p>=0) and np.all(p<=1)
  replay=r['replay'][split];assert replay['queries']==n and replay['max_original_logit_error']<1e-5
  maximum=max(maximum,replay['max_original_logit_error']);total+=n;scores[split]=metric
  if seed:
   reference=np.load(P/'evidence/l128/paper/seed-0/predictions.npz')
   for suffix in ['target','entity','time']:np.testing.assert_array_equal(pred[split+'_'+suffix],reference[split+'_'+suffix])
 t=json.loads((root/'temporal-audit.json').read_text());assert t['status']=='PASS'
 for split,n in [('train',114110),('val',6226),('test',702)]:assert t['splits'][split]['queries']==n
 assert t['audit_sha256']==hashlib.sha256((P/'relkit/batch_audit_l123.py').read_bytes()).hexdigest()
 learner_records.append(dict(seed=seed,status=done['status'],run_uuid=done['run_uuid'],epochs=r['epochs'],**scores))
 rows.append(dict(seed=seed,selected_epoch=r['selected_epoch'],**scores));cost+=done['resource_usd']
cost+=json.loads((P/'evidence/l128/pilot/seed-100/completed.json').read_text())['resource_usd']
aggregate=summarize_seeds(learner_records,contract["seeds"])
metrics={}
for split,target in [('val',.7136),('test',.7262)]:
 values=[r[split] for r in rows];mean=statistics.mean(values)
 assert mean==aggregate[split]['mean'] and statistics.stdev(values)==aggregate[split]['sample_sd']
 metrics[split]=dict(mean=mean,sample_sd=statistics.stdev(values),target=target,delta=mean-target,descriptive_tolerance=.02,verdict='CLOSE' if abs(mean-target)<=.02 else 'OUTSIDE_TOLERANCE')
s=dict(status='COMPLETE',experiment='RelBench v1 Table 6 driver-dnf RDL; historical-label reconstruction',lesson=128,fresh_execution=True,seeds=rows,metrics=metrics,evaluated_queries=total,maximum_original_output_error=maximum,worker_resource_usd=cost,all_query_identities='MATCH_ACROSS_SEEDS',all_epochs='COMPLETE',historical_identity='NOT_ESTABLISHED',whole_paper_parity='NOT_ESTABLISHED',live_colab='NOT_CHECKED',learner_status='PENDING_WRITTEN_DEFENSE',deviations=['Source fanout [128,64] versus paper table 128','Declared seeds 0-4; historical seed identities unavailable','Type inference fixed seed42 before reset to training seed','Current pinned runtime and GloVe revision; historical package/random-state identity unavailable','Statistics fit to database up to test cutoff, not train-only','Real ingestion and mutable-feature histories absent','Current DNF archive differs from old registry; labels reconstructed from exact pre-flip SQL','Historical archive bytes and ordering unavailable; current query order retained'])
s['failed_pilot_resource_bound_usd']=budget['failed_pilot_resource_bound_usd']
s['temporal_audit']={'status':'PASS','audited_query_occurrences':sum(sum(v['queries'] for v in json.loads((P/f'evidence/l128/paper/seed-{seed}/temporal-audit.json').read_text())['splits'].values()) for seed in range(5)),'scope':'Every training, epoch-validation and final-evaluation batch; node times, edge identity, query isolation'}
budget.update(status='COMPLETE',recorded_worker_resource_usd=cost,billing_total='NOT_ITEMIZED',failed_pilot_resource_bound_usd=budget['failed_pilot_resource_bound_usd'])
(P/'_budget_l128.json').write_text(json.dumps(budget,indent=2)+'\n')
(P/'evidence/l128/summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
