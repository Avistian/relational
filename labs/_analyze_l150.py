"""Independently summarize every complete track; keep search/replay/validation separate."""
import hashlib,json,math,statistics
from pathlib import Path
import numpy as np
from relkit.checkpoint_l150 import select_candidate,summarize_track,checkpoint_verdict
P=Path(__file__).resolve().parent;E=P/'evidence/l150'
def read(p):return json.loads(p.read_text())
frozen=read(E/'frozen.json');assert select_candidate(frozen['candidates'])==frozen['lr']
for p,h in frozen['files'].items():assert hashlib.sha256((P.parent/p).read_bytes()).hexdigest()==h
tracks={};count=0;histories={};arrays={};files={};records=[]
for track,seeds,prefix in [('reference',range(5),'ref'),('search',[1,3,5],'search'),('selected',range(10,15),'selected'),('replay',[42],'replay')]:
 rows=[]
 for s in seeds:
  phase='replay-compatible' if track=='replay' else (f'search-{s:03d}' if track=='search' else f'{prefix}-{s}')
  r=read(E/phase/'result.json');a=np.load(E/phase/'predictions.npz');splits=['val'] if track=='search' else ['val','test']
  assert r['track']==track and set(r['scores'])==set(splits) and r['status']=='COMPLETE'
  if track in ['reference','selected','search']:
   assert len(r['history'])==10 and all(h['queries']==7453 for h in r['history'])
   assert r['best_epoch']==min(r['history'],key=lambda h:h['val_mae'])['epoch']
  if track=='selected':assert r['lr']==frozen['lr']
  if track=='reference':assert r['lr']==.005
  assert r['temporal_audit']['future_violations']==0
  row=dict(phase=phase,seed=r['seed'],track=track,lr=r['lr'],complete=True,epochs=r['epochs'],selection_mae=r['selection_mae'],nonfinite_entries=sum(r['nonfinite_gradients'].values()))
  for split in splits:
   keys=list(zip(a[split+'_entity'],a[split+'_time']));assert len(set(keys))==len(keys)
   score=math.fsum(abs(float(y)-float(p)) for y,p in zip(a[split+'_target'],a[split+'_pred']))/len(keys)
   assert abs(score-r['scores'][split]['mae'])<1e-12;count+=len(keys);row[split+'_mae']=score
  rows.append(row);records.append(row);histories[phase]=r['history']
  for k in a.files:arrays[phase+'_'+k]=a[k]
  for name in ['result.json','predictions.npz','sampled_trace.npz']:files[phase+'/'+name]=hashlib.sha256((E/phase/name).read_bytes()).hexdigest()
 if track in ['reference','selected']:
  summary=summarize_track(rows,track,list(seeds));summary['val_mean']=statistics.mean(r['val_mae'] for r in rows);summary['val_sample_sd']=statistics.stdev(r['val_mae'] for r in rows)
  summary['gates']=checkpoint_verdict(True,summary['mean'],False,False);tracks[track]=summary
assert count==15346
np.savez_compressed(E/'portable.npz',**arrays)
summary=dict(status='COMPLETE_APPROVED_EXPERIMENT',tracks=tracks,records=records,search=frozen,predictions=count,primary_training_fits=10,search_fits=3,replay_fits=1,histories=histories,files=files,paper_target=3.798,tolerance=.20,competitive='NOT_ESTABLISHED',historical_identity='NOT_ESTABLISHED',whole_paper='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')
(E/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['histories','files']},indent=2))
