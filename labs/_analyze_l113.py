"""Reconstruct selected metrics from predictions and label IDs, not logged scores."""
import json,hashlib
from pathlib import Path
import numpy as np
from ogb.nodeproppred import Evaluator
P=Path(__file__).resolve().parent;D=P/'evidence/l113';records=[];labels_path=D/'labels-splits.npz';source=json.loads((P/'_sources_l113.json').read_text())['source_sha256']
if labels_path.exists():
 labels=np.load(labels_path);y=labels['y'];evaluator=Evaluator(name='ogbn-products')
 for seed in range(10):
  root=D/'paper'/f'seed-{seed}'
  if not (root/'identity.json').exists() or not (root/'result.json').exists():continue
  r=json.loads((root/'result.json').read_text());identity=json.loads((root/'identity.json').read_text());assert identity['source_sha256']==source
  if r['status']!='COMPLETE':continue
  assert len(r['epochs'])==50 and all(a['finished'] for a in r['epochs']) and [a['epoch'] for a in r['history']]==list(range(20,51,5))
  assert all(a['training_examples']==196615 for a in r['epochs'])
  selected=max(r['history'],key=lambda x:x['valid']);assert selected==r['selected']
  pred=np.load(root/'predictions.npz')['pred'];assert pred.shape==y.shape
  scores={}
  for key in ['train','valid','test']:
   idx=labels[key];exact=float(np.mean(pred[idx]==y[idx]));official=evaluator.eval({'y_true':y[idx,None],'y_pred':pred[idx,None]})['acc'];assert exact==official
   assert abs(exact-selected[key])<1e-7;scores[key]=exact*100
  records.append({'seed':seed,'selected_epoch':selected['epoch'],'valid_percent':scores['valid'],'test_percent':scores['test'],'seconds':r['seconds'],'peak_cuda_allocated_bytes':r['peak_cuda_allocated_bytes'],'resource_usd':identity['resource_usd']})
complete=len(records)==10
r={'status':'COMPLETE' if complete else ('INCOMPLETE' if records else 'NOT_RUN'),'seeds':records,'source_sha256':source,'historical_identity':'NOT_ESTABLISHED','whole_paper':'NOT_ESTABLISHED','learner':'PENDING_WRITTEN_DEFENSE','interpretation':'All ten schedules and independent metric reconstructions complete.' if complete else 'The ten-run published experiment is not complete. Partial runs and pilots do not establish its mean or seed variability.'}
if complete:
 r['summary']={}
 for key,target in [('valid',92.12),('test',78.97)]:
  a=np.array([x[key+'_percent'] for x in records]);gap=float(a.mean()-target);r['summary'][key]={'mean_percent':float(a.mean()),'sample_sd_pp':float(a.std(ddof=1)),'target_percent':target,'gap_pp':gap,'verdict':'CLOSE' if abs(gap)<=.5 else 'OUTSIDE_TOLERANCE'}
 r['interpretation']='; '.join(f"{k}: {a['mean_percent']:.4f}% ± {a['sample_sd_pp']:.4f}pp (sample seed SD), {a['verdict']}" for k,a in r['summary'].items())+'. Historical identity and whole-paper parity are not established.'
(D/'summary.json').write_text(json.dumps(r,indent=2));print(r)
