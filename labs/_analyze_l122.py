"""Independent scalar MAE, query identity, checkpoint and protocol checks."""
import hashlib,json,math,statistics
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
rows=[];maximum=0.;total=0;cost=0.;schema=None
uuids=set()
budget=json.loads((P/'_budget_l122.json').read_text())
for seed in range(5):
 root=P/f'evidence/l122/paper/seed-{seed}'
 r=json.loads((root/'result.json').read_text());a=json.loads((root/'audit.json').read_text());done=json.loads((root/'completed.json').read_text())
 assert done['lesson']==122 and done['seed']==seed and done['status']=='COMPLETE'
 assert done['started_utc']>=budget['reservations'][1]['utc'] and done['finished_utc']>done['started_utc']
 assert done['run_uuid'] not in uuids
 uuids.add(done['run_uuid'])
 assert done['runner_sha256']==hashlib.sha256((P/'_run_l117.py').read_bytes()).hexdigest()
 assert r['seed']==seed
 assert r['epochs']==10 and len(r['trace'])==10 and all(v['train_queries']==7453 for v in r['trace'])
 assert r['selected_epoch']==min(r['trace'],key=lambda x:x['val_mae'])['epoch']
 assert r['checkpoint_sha256']==hashlib.sha256((P/f'results/l122/paper/seed-{seed}/selected.pt').read_bytes()).hexdigest()
 assert a['source_sha256']==hashlib.sha256((P/'relkit/rdl_l117.py').read_bytes()).hexdigest()
 assert sum(a['rows'].values())==74063
 current=(a['stypes'],a['rows'],a['edges'],a['packages'],list(a['rows']))
 if schema is None:schema=current
 else:assert schema==current,'Preprocessing/runtime/table ordering differed across runs'
 pred=np.load(root/'predictions.npz');scores={}
 for split,n in [('val',499),('test',760)]:
  y=pred[split+'_target'];p=pred[split+'_pred'];assert len(y)==n and np.isfinite(p).all()
  metric=math.fsum(abs(float(x)-float(z)) for x,z in zip(p,y))/n
  assert abs(metric-r['scores'][split])<1e-12
  assert np.all(p>=r['clamp'][0]-1e-5) and np.all(p<=r['clamp'][1]+1e-5)
  replay=r['replay'][split];assert replay['queries']==n and replay['max_original_logit_error']<1e-5
  maximum=max(maximum,replay['max_original_logit_error']);total+=n;scores[split]=metric
  if seed:
   reference=np.load(P/'evidence/l122/paper/seed-0/predictions.npz')
   for suffix in ['target','entity','time']:np.testing.assert_array_equal(pred[split+'_'+suffix],reference[split+'_'+suffix])
 rows.append(dict(seed=seed,selected_epoch=r['selected_epoch'],**scores));cost+=done['resource_usd']
cost+=json.loads((P/'evidence/l122/pilot/seed-100/completed.json').read_text())['resource_usd']
metrics={}
for split,target in [('val',3.193),('test',4.022)]:
 values=[r[split] for r in rows];mean=statistics.mean(values)
 metrics[split]=dict(mean=mean,sample_sd=statistics.stdev(values),target=target,delta=mean-target,descriptive_tolerance=.2,verdict='CLOSE' if abs(mean-target)<=.2 else 'OUTSIDE_TOLERANCE')
s=dict(status='COMPLETE',experiment='RelBench v1 Table 7 F1 driver-position RDL; released-protocol replay',lesson=122,fresh_execution=True,seeds=rows,metrics=metrics,evaluated_queries=total,maximum_original_output_error=maximum,worker_resource_usd=cost,all_query_identities='MATCH_ACROSS_SEEDS',all_epochs='COMPLETE',historical_identity='NOT_ESTABLISHED',whole_paper_parity='NOT_ESTABLISHED',live_colab='NOT_CHECKED',learner_status='PENDING_WRITTEN_DEFENSE',deviations=['Source fanout [128,64] versus paper table 128','Declared seeds 0-4; historical seed identities unavailable','Type inference fixed seed42 before reset to training seed','Current pinned runtime and GloVe revision; historical package/random-state identity unavailable','Statistics fit to database up to test cutoff, not train-only','Real ingestion and mutable-feature histories absent'])
(P/'evidence/l122/summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
