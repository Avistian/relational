"""Fail-closed independent metrics and identity audit; never substitute partial runs."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
P=Path(__file__).resolve().parent;E=P/'evidence/l107'
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def ap(y,p):
 order=np.argsort(-p,kind='stable');y=np.asarray(y)[order];p=p[order]
 ends=np.r_[np.flatnonzero(np.diff(p)),len(y)-1];tp=np.cumsum(y)[ends];return float(np.sum(np.diff(np.r_[0,tp])*tp/(ends+1))/tp[-1])
def auc(y,p):
 order=np.argsort(p,kind='stable');y=np.asarray(y)[order];p=p[order];bounds=np.r_[0,np.flatnonzero(np.diff(p))+1,len(p)];credit=0.;neg=0
 for start,end in zip(bounds[:-1],bounds[1:]):
  pos=y[start:end].sum();n=end-start-pos;credit+=pos*(neg+.5*n);neg+=n
 return float(credit/(y.sum()*(len(y)-y.sum())))
def mrr(pairs,y,p,n):
 # Literal source-node loop, independently constructing ranks.
 values=[]
 for u in range(n):
  ids=np.flatnonzero(pairs[:,0]==u);score=np.zeros(n,dtype=p.dtype);truth=np.zeros(n,dtype=bool);score[pairs[ids,1]]=p[ids];truth[pairs[ids,1]]=y[ids]
  if truth.any():
   ordered=np.flip(np.argsort(score));ranks=np.flatnonzero(truth[ordered])+1;values.append(np.mean(1/ranks))
 return float(np.mean(values))

def tied_mrr_interval(pairs,y,p,n):
 # All possible rankings inside equal-score groups; no arbitrary tie convention.
 score=np.zeros((n,n),dtype=p.dtype);truth=np.zeros((n,n),dtype=bool)
 score[pairs[:,0],pairs[:,1]]=p;truth[pairs[:,0],pairs[:,1]]=y
 harmonic=np.r_[0.,np.cumsum(1/np.arange(1,n+1))];interval=[]
 for values,labels in zip(score,truth):
  if not labels.any():continue
  order=np.argsort(-values,kind='stable');values=values[order];labels=labels[order]
  bounds=np.r_[0,np.flatnonzero(np.diff(values))+1,n]
  counts=np.diff(np.r_[0,np.cumsum(labels)[bounds[1:]-1]])
  lo=np.sum(harmonic[bounds[1:]]-harmonic[bounds[1:]-counts])/labels.sum()
  hi=np.sum(harmonic[bounds[:-1]+counts]-harmonic[bounds[:-1]])/labels.sum()
  interval.append([lo,hi])
 return np.mean(interval,axis=0).tolist()

def analyze():
 out={'status':'RUNNING','wiki':{},'sbm':{},'artifacts':{},'source_hashes':{n:sha(P/'relkit'/n) for n in ['snapshot_l107.py','wiki_snapshot_l107.py','sbm_l107.py','tgn_l102.py']},'learner_status':'PENDING_WRITTEN_DEFENSE','historical_identity':'NOT_ESTABLISHED','other_paper_datasets':'NOT_RUN'}
 keys={};questions={};errors=[];local_tie_differences=[];tie_audits=[]
 for arm in ['3600','86400','tgn']:
  records=[]
  for seed in range(3):
   root=E/'wiki'/arm/f'seed-{seed}';r=json.loads((root/'result.json').read_text());identity=json.loads((root/'identity.json').read_text())
   assert r['status']=='COMPLETE' and r['epochs']==10
   out['wiki_counts']=identity['audit']['counts']
   assert identity['torch']=='2.8.0+cu128' or identity['torch'].startswith('2.8.0')
   for name in ['snapshot_l107.py','wiki_snapshot_l107.py','tgn_l102.py']:assert identity[name]==out['source_hashes'][name]
   z=np.load(root/'predictions.npz');q=np.load(root/'questions.npz');idx=np.flatnonzero(q['split']==2);expected=np.sort(q['e'][idx]);assert len(idx)==23621
   for label in [0,1]:np.testing.assert_array_equal(np.sort(z['edge'][z['y']==label]),expected)
   for name in q.files:
    key=(seed,name)
    if key in questions:np.testing.assert_array_equal(questions[key],q[name])
    else:questions[key]=q[name]
   key=np.lexsort((z['y'],z['edge']));canonical=np.stack([z['edge'][key],z['y'][key]],1)
   if seed in keys:np.testing.assert_array_equal(keys[seed],canonical)
   else:keys[seed]=canonical
   metrics={'ap':ap(z['y'],z['prob']),'auc':auc(z['y'],z['prob'])}
   for m in metrics:errors.append(abs(metrics[m]-r['test'][m]));assert errors[-1]<1e-10
   assert r['selected_epoch']==max(range(10),key=lambda i:r['trace'][i]['val']['ap'])
   records.append({k:r[k] for k in ['seed','selected_epoch','test','epochs','seconds']})
  out['wiki'][arm]={'runs':records,'mean':{m:float(np.mean([r['test'][m] for r in records])) for m in ['ap','auc']},'sd':{m:float(np.std([r['test'][m] for r in records],ddof=1)) for m in ['ap','auc']}}
 for variant in ['H','O']:
  root=E/'sbm'/variant
  if not (root/'result.json').exists():continue
  r=json.loads((root/'result.json').read_text());identity=json.loads((root/'identity.json').read_text());assert r['status']=='COMPLETE',variant+' incomplete'
  replay=json.loads((root/'source_replay.json').read_text());assert replay['status']=='PASS' and len(replay['metric_rows'])==10
  for name in ['snapshot_l107.py','sbm_l107.py']:assert identity[name]==out['source_hashes'][name]
  selected=max((x for x in r['trace'] if 'valid' in x),key=lambda x:x['valid']['map']);assert selected['epoch']==r['selected_epoch']
  metrics=[]
  for t in range(39,49):
   z=np.load(root/f'test-{t}.npz');assert len(z['y'])==1000000;ids=z['pairs'][:,0].astype(np.int64)*1000+z['pairs'][:,1];np.testing.assert_array_equal(np.sort(ids),np.arange(1000000))
   values={'map':ap(z['y'],z['prob']),'mrr':mrr(z['pairs'],z['y'],z['prob'],1000)}
   row=next(x for x in r['selected']['per_snapshot'] if x['t']==t)
   errors.append(abs(values['map']-row['map']));assert errors[-1]<1e-10
   interval=tied_mrr_interval(z['pairs'],z['y'],z['prob'],1000)
   assert interval[0]-1e-12<=row['mrr']<=interval[1]+1e-12
   assert interval[0]-1e-12<=values['mrr']<=interval[1]+1e-12
   local_tie_differences.append(abs(values['mrr']-row['mrr']))
   tie_audits.append({'variant':variant,'t':t,'mrr_interval':interval,'local_mrr':values['mrr'],'pinned_mrr':row['mrr']})
   pinned=next(x for x in replay['metric_rows'] if x['t']==t)
   for method in ['independent','original']:
    for m in values:errors.append(abs(pinned[method][m]-row[m]));assert errors[-1]<1e-10
   metrics.append(pinned['independent'])
  mean={m:float(np.mean([x[m] for x in metrics])) for m in ['map','mrr']}
  targets=json.loads((P/'_sources_l107.json').read_text())['targets'][variant];tol={'map':.015,'mrr':.003}
  out['sbm'][variant]={'epochs':r['epochs'],'selected_epoch':r['selected_epoch'],'seed':r['seed'],'test':mean,'target':targets,'verdict':{m:'CLOSE' if abs(mean[m]-targets[m])<=tol[m] else 'FAIL' for m in mean},'seconds':r['seconds']}
 for file in E.rglob('*'):
  if file.is_file():out['artifacts'][str(file.relative_to(E))]=sha(file)
 out['checks']={'independent_metric_max_error':max(errors),'matched_question_and_candidate_arrays':'EXACT','wiki_test_predictions_checked':9*47242,'SBM_test_predictions_checked':len(out['sbm'])*10000000,'best_validation_selection':'PASS','local_runtime_max_MRR_tie_difference':max(local_tie_differences,default=0),'MRR_tie_intervals':tie_audits,'pinned_runtime_original_and_independent_metrics':'PASS' if len(out['sbm'])==2 else 'PARTIAL'}
 if len(out['sbm'])==2:out['status']='COMPLETE'
 (P/'_analysis_l107_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['artifacts','source_hashes']},indent=2));return out
if __name__=='__main__':analyze()
