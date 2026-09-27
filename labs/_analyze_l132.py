"""Separate preparation, feasibility pilots and complete five-seed benchmark evidence."""
import json,statistics
from pathlib import Path
import numpy as np,pandas as pd
from relkit.identity_l132 import mean_average_precision
P=Path(__file__).resolve().parent;E=P/'evidence/l132';b=json.loads((P/'_budget_l132.json').read_text())
r=dict(status='NOT_RUN',explanation='The full ten-fit benchmark has not been dispatched.',pilots=[],failures=[],metrics={},paper_targets={'sage':{'val':.0312,'test':.0289},'idgnn':{'val':.1133,'test':.1136}},descriptive_mean_tolerance=.005,whole_paper='NOT_ESTABLISHED',historical_identity='NOT_ESTABLISHED',live_colab='NOT_CHECKED',learner='PENDING_WRITTEN_DEFENSE')
if (E/'prepared/prepared.json').exists():r['preparation']=json.loads((E/'prepared/prepared.json').read_text())
costs=[]
if (E/'preparation-cost.json').exists():costs.append(json.loads((E/'preparation-cost.json').read_text())['resource_usd'])
for done in E.glob('*/**/completed.json'):
 d=json.loads(done.read_text());costs.append(d['resource_usd'])
 if d['status']!='COMPLETE':r['failures'].append(dict(path=str(done.parent.relative_to(E)),status=d['status'],error=(done.parent/'error.txt').read_text() if (done.parent/'error.txt').exists() else 'No error artifact'))
for p in E.glob('pilot/*/result.json'):r['pilots'].append(json.loads(p.read_text()))
all_rows=[];count=0
for variant in ['sage','idgnn']:
 rows=[]
 for seed in range(5):
  root=E/f'paper/{variant}-{seed}'
  if not (root/'result.json').exists():continue
  result=json.loads((root/'result.json').read_text());assert result['epochs']==20 and len(result['trace'])==20
  vals=[x['val'] for x in result['trace']];best=max(vals);selected=[i+1 for i,v in enumerate(vals) if v==best]
  assert result['selected_epoch']==(selected[-1] if variant=='sage' else selected[0])
  a=np.load(root/'predictions.npz')
  for split in ['val','test']:
   df=pd.read_parquet(E/f'prepared/{split}.parquet');pred=a[split+'_pred']
   assert len(pred)==len(df) and len(set(zip(a[split+'_source'],a[split+'_time'])))==len(df)
   np.testing.assert_array_equal(a[split+'_source'],df.condition_id);np.testing.assert_array_equal(a[split+'_time'],df.timestamp.astype('int64'))
   assert (pred>=0).all() and (pred<r['preparation']['num_dst']).all()
   score=mean_average_precision(pred,df.sponsor_id.tolist(),10);assert abs(score-result['scores'][split])<1e-10;count+=len(pred)
  rows.append(result);all_rows.append(result)
 if len(rows)==5:
  r['metrics'][variant]={}
  for split in ['val','test']:
   values=[x['scores'][split] for x in rows];mean=statistics.mean(values);target=r['paper_targets'][variant][split]
   r['metrics'][variant][split]=dict(mean=mean,sample_sd=statistics.stdev(values),target=target,verdict='CLOSE' if abs(mean-target)<=.005 else 'NOT_CLOSE')
r.update(completed_full_fits=len(all_rows),full_predictions_rescored=count,recorded_worker_resource_usd=sum(costs),billing='NOT_ITEMIZED',budget_usd=10,reservations=b['reservations'],pilot_projection=b.get('pilot_projection'))
if len(all_rows)==10:r.update(status='COMPLETE',explanation='Ten fresh full-data 20-epoch fits completed and every final ranked prediction was independently rescored.')
elif r['failures']:r.update(status='INCOMPLETE',explanation='Feasibility execution failed before a complete ten-fit comparison. Failure evidence is retained; no full paper-result claim is made.')
elif b.get('pilot_projection') and not b['pilot_approved_for_full']:r.update(status='INCOMPLETE',explanation='Measured pilot timing exceeds the per-fit/aggregate budget gate. Full schedules remain executable but were not dispatched under the USD10 cap.')
elif len(all_rows)>0:r.update(status='INCOMPLETE',explanation='Only some of the ten required full fits completed. No complete benchmark comparison is claimed.')
(E/'summary.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k not in ['pilots','failures','preparation','reservations']})
