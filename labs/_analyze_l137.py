"""Independent fresh-fit audit; no comparison to a different leaderboard run."""
import hashlib,json,math,statistics
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l137';budget=json.loads((P/'_budget_l137.json').read_text())
for name,digest in budget['source_hashes'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
records=[];resource=0;count=0;max_error=0.;hashes={}
for phase,seeds in [('pilot',[999]),('final',range(5))]:
 for seed in seeds:
  root=E/phase/'lr005-full'/f'seed-{seed}';r=json.loads((root/'result.json').read_text());done=json.loads((root/'completed.json').read_text());a=np.load(root/'predictions.npz')
  assert done['status']=='COMPLETE' and r['seed']==seed
  for name,digest in done['source_hashes'].items():assert digest==budget['source_hashes']['labs/'+name]
  epochs=1 if phase=='pilot' else 10
  assert len(r['trace'])==epochs and all(t['train_queries']==7453 for t in r['trace'])
  assert r['selected_epoch']==min(r['trace'],key=lambda t:t['val_mae'])['epoch']
  assert set(r['scores'])==({'val'} if phase=='pilot' else {'val','test'})
  for split,score in r['scores'].items():
   n=499 if split=='val' else 760
   assert len(a[split+'_pred'])==n==len(set(zip(a[split+'_entity'],a[split+'_time'])))
   measured=math.fsum(abs(float(x)-float(y)) for x,y in zip(a[split+'_pred'],a[split+'_target']))/n
   assert abs(measured-score)<1e-12;count+=n
   max_error=max(max_error,r['replay'][split]['max_original_logit_error'])
  ck=P/'results/l137'/phase/'lr005-full'/f'seed-{seed}'/'selected.pt';assert hashlib.sha256(ck.read_bytes()).hexdigest()==r['checkpoint_sha256']
  assert r['temporal_audit']['train']['queries']==7453*epochs
  resource+=done['resource_usd']
  if phase=='final':records.append(r)
  for path in root.iterdir():hashes[str(path.relative_to(R))]=hashlib.sha256(path.read_bytes()).hexdigest()
metrics={}
for split,target in [('val',3.193),('test',4.022)]:
 values=[r['scores'][split] for r in records];mean=statistics.mean(values)
 metrics[split]=dict(values=values,mean=mean,sample_sd=statistics.stdev(values),paper_target=target,absolute_delta=abs(mean-target),verdict='CLOSE' if abs(mean-target)<=.2 else 'OUTSIDE_TOLERANCE')
s=dict(status='COMPLETE_SELECTED_TRAINING',seeds=list(range(5)),epochs=10,train_queries_per_epoch=7453,metrics=metrics,descriptive_tolerance=.2,predictions_independently_rescored=count,maximum_original_output_error=max_error,worker_resource_usd=resource,reserved_worker_upper_usd=sum(r['upper_usd'] for r in budget['reservations']),files=hashes,historical_identity='NOT_ESTABLISHED',current_gnn_entry_training='NOT_ESTABLISHED',whole_paper='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')
(E/'training.json').write_text(json.dumps(s,indent=2));print(json.dumps({k:v for k,v in s.items() if k!='files'},indent=2))
