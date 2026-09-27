"""Independently rescore every saved prediction and compare frozen paired conditions."""
import datetime,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
from scipy.stats import t
from relkit.tuning_l135 import paired_differences,select_configuration
P=Path(__file__).parent;R=P.parent;E=P/'evidence/l135'
protocol=json.loads((P/'_protocol_l135.json').read_text());frozen=json.loads((E/'frozen.json').read_text());budget=json.loads((P/'_budget_l135.json').read_text())
assert hashlib.sha256((P/'_protocol_l135.json').read_bytes()).hexdigest()==frozen['protocol_sha256']
for name,digest in budget['source_hashes'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
for name,digest in frozen['search_hashes'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
entry=next(r for r in budget['reservations'] if r['phase']=='final')
assert entry['freeze_sha256']==hashlib.sha256((E/'frozen.json').read_bytes()).hexdigest()
assert frozen['frozen_utc']<entry['utc']
search=[];final={};predictions=0;audited=0;max_error=0.;files={};resource=0.;rows=[]
for phase in ['pilot','search','final']:
 for path in sorted((E/phase).glob('*/seed-*/result.json')):
  root=path.parent;r=json.loads(path.read_text());done=json.loads((root/'completed.json').read_text());resource+=done['resource_usd']
  assert done['status']=='COMPLETE';assert r['configuration']['id']==root.parent.name
  for name,digest in done['source_hashes'].items():assert digest==budget['source_hashes']['labs/'+name],name
  assert len(r['trace'])==r['epochs'] and all(x['train_queries']==7453 for x in r['trace'])
  selected=min(r['trace'],key=lambda x:x['val_mae']);assert r['selected_epoch']==selected['epoch'] and r['selection_mae']==selected['val_mae']
  splits=['val','test'] if phase=='final' else ['val'];assert set(r['scores'])==set(splits) and r['evaluate_test']==(phase=='final')
  arrays=np.load(root/'predictions.npz')
  assert set(arrays.files)=={s+'_'+v for s in splits for v in ['pred','target','entity','time']}
  ck=P/'results/l135'/phase/root.parent.name/root.name/'selected.pt'
  assert hashlib.sha256(ck.read_bytes()).hexdigest()==r['checkpoint_sha256']
  for split in splits:
   n=499 if split=='val' else 760
   assert len(set(zip(arrays[split+'_entity'],arrays[split+'_time'])))==n
   score=math.fsum(abs(float(x)-float(y)) for x,y in zip(arrays[split+'_pred'],arrays[split+'_target']))/n
   assert abs(score-r['scores'][split])<1e-12
   max_error=max(max_error,r['replay'][split]['max_original_logit_error']);predictions+=n
  audit=r['temporal_audit'];assert audit['train']['queries']==7453*r['epochs'];assert audit['val']['queries']==499*(r['epochs']+1)
  if phase=='final':assert audit['test']['queries']==760
  audited+=sum(a['queries'] for a in audit.values())
  if phase=='search':search.append(dict(config=root.parent.name,seed=r['seed'],selection_mae=r['selection_mae']))
  if phase=='final':final.setdefault(root.parent.name,[]).append(r)
  rows.append(dict(phase=phase,config=root.parent.name,seed=r['seed'],seconds=done['seconds'],training_seconds=r['seconds'],resource_usd=done['resource_usd']))
  for file in root.iterdir():files[str(file.relative_to(R))]=hashlib.sha256(file.read_bytes()).hexdigest()
assert len(search)==12
assert select_configuration(search,[c['id'] for c in protocol['configurations']],protocol['search_seeds'])['winner']==frozen['winner']
assert set(final)=={protocol['default'],frozen['winner']}
metrics={}
for c,records in final.items():
 assert sorted(r['seed'] for r in records)==protocol['final_seeds']
 metrics[c]={}
 for split in ['val','test']:
  values=[r['scores'][split] for r in records]
  m=dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values),values=values)
  if c==protocol['default']:
   target=protocol['paper_targets'][split];m.update(target=target,verdict='CLOSE' if abs(m['mean']-target)<=protocol['descriptive_tolerance'] else 'OUTSIDE_TOLERANCE')
  metrics[c][split]=m
pair=paired_differences([dict(seed=r['seed'],mae=r['scores']['test']) for r in final[protocol['default']]],[dict(seed=r['seed'],mae=r['scores']['test']) for r in final[frozen['winner']]])
half=float(t.ppf(.975,len(pair['seeds'])-1))*pair['sample_sd']/math.sqrt(len(pair['seeds']))
pair['conditional_t95']=[pair['mean_difference']-half,pair['mean_difference']+half]
pair['scope']='Conditional on fixed task, split and selected winner; excludes search and dataset uncertainty'
summary=dict(status='COMPLETE',winner=frozen['winner'],default=protocol['default'],search_trials=12,final_fits=sum(map(len,final.values())),metrics=metrics,paired_test=pair,predictions_independently_rescored=predictions,audited_query_occurrences=audited,maximum_original_output_error=max_error,worker_resource_usd=resource,reserved_worker_upper_usd=sum(x['upper_usd'] for x in budget['reservations'] if x['phase'] in ['pilot','search','final']),billing_total='NOT_ITEMIZED',worker_rows=rows,files=files,historical_training_identity='NOT_ESTABLISHED',whole_paper='NOT_RUN',live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(E/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['files','worker_rows']},indent=2))
