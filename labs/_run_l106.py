"""Full original-candidate replay. No partial cache/resume; complete evidence only."""
import argparse,hashlib,importlib.util,json,sys,time,platform,random,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'sources/l106'))
import edge_bank_baseline as upstream
from edge_sampler import RandEdgeSampler,RandEdgeSampler_adversarial
spec=importlib.util.spec_from_file_location('visible',P/'relkit/edgebank_l106.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(raw,out):
 start=time.perf_counter();out.mkdir(parents=True,exist_ok=True)
 source=json.loads((P/'_sources_l106.json').read_text())
 for f in source['files']:assert sha(P.parent/f['path'])==f['sha256']
 events=m.load_events(raw);hi,ti,held,cuts=m.released_split(events)
 # Compare complete split to original get_data, with ONLY set->tuple compatibility.
 with tempfile.TemporaryDirectory() as tmp:
  d=Path(tmp);u,v,t=events.T
  pd.DataFrame({'u':u.astype(int),'i':v.astype(int),'ts':t,'idx':np.arange(len(t)),'label':np.zeros(len(t))}).to_csv(d/'ml_wikipedia.csv',index=False)
  np.save(d/'ml_wikipedia.npy',np.zeros((1,1)));np.save(d/'ml_wikipedia_node.npy',np.zeros((1,1)))
  old_sample=random.sample
  random.sample=lambda population,k:old_sample(tuple(population) if isinstance(population,set) else population,k)
  try:_,_,full,tr,va,te,*_=upstream.get_data(str(d),'wikipedia',.15,.15)
  finally:random.sample=old_sample
  np.testing.assert_array_equal(hi,np.r_[tr.edge_idxs,va.edge_idxs]);np.testing.assert_array_equal(ti,te.edge_idxs)
 np.savez_compressed(out/'split.npz',history_ids=hi,test_ids=ti,held_nodes=held)
 report={'status':'RUNNING','source_commit':source['commit'],'raw_sha256':m.DATA_SHA,'events':len(events),'history':len(hi),'test':len(ti),'cut_times':cuts,'held_nodes':len(held),'split_source_parity':'PASS','python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'conditions':{},'budget_usd':0,'full_paper':'NOT_RUN','historical_identity':'NOT_ESTABLISHED'}
 for strategy in ['rnd','hist_nre','induc_nre']:
  print('Constructing original sampler',strategy,flush=True)
  u,v,t=events.T;u=u.astype(np.int64);v=v.astype(np.int64)
  sampler=RandEdgeSampler(u,v,seed=2) if strategy=='rnd' else RandEdgeSampler_adversarial(u,v,t,va.timestamps[-1],strategy,seed=2)
  all_neg=[]
  for run_idx in range(5):
   sampler.reset_random_state();batches=[]
   for s in range(0,len(ti),200):
    ids=ti[s:s+200]
    if strategy=='rnd':
     a,b=sampler.sample(len(ids),u[ids],v[ids]);a=u[ids] # preserve source's post-exclusion replacement
    else:a,b=sampler.sample(len(ids),u[ids],v[ids],t[ids[0]],t[ids[-1]])
    batches.append(np.column_stack([a,b]).astype(np.int64))
    if s%6000==0:print(strategy,run_idx,'events',s,'seconds',round(time.perf_counter()-start),flush=True)
    if time.perf_counter()-start>2700:raise TimeoutError('45 minute local replay cutoff; no complete report emitted')
   all_neg.append(np.concatenate(batches))
  del sampler
  negative=np.asarray(all_neg);np.savez_compressed(out/f'negatives-{strategy}.npz',edges=negative)
  rows,pred=m.replay(events,hi,ti,negative)
  # Independent original-model and sklearn metric oracle on EVERY batch/run/memory.
  from sklearn.metrics import average_precision_score,roc_auc_score
  for ri in range(5):
   for bi,s in enumerate(range(0,len(ti),200)):
    ids=ti[s:s+200];h=events[np.r_[hi,ti[:s]]];pos=events[ids,:2];neg=negative[ri,s:s+len(ids)]
    data=upstream.Data(h[:,0],h[:,1],h[:,2],np.arange(len(h)),np.zeros(len(h)))
    for mode in ['unlimited','window']:
     pp,nn=upstream.edge_bank_link_pred_end_to_end(data,(pos[:,0],pos[:,1]),(neg[:,0],neg[:,1]),{'m_mode':'unlim_mem' if mode=='unlimited' else 'time_window','w_mode':'fixed'})
     np.testing.assert_array_equal(np.stack([pp,nn],axis=1),pred[ri][mode][bi])
     y=np.r_[np.ones(len(pp)),np.zeros(len(nn))];ss=np.r_[pp,nn]
     assert np.allclose([average_precision_score(y,ss),roc_auc_score(y,ss)],rows[ri][mode]['batch_metrics'][bi],atol=1e-14)
  np.savez_compressed(out/f'predictions-{strategy}.npz',**{mode:np.array([np.concatenate(x[mode]) for x in pred]) for mode in ['unlimited','window']})
  summary={}
  for mode in ['unlimited','window']:
   vals=np.array([r[mode]['batch_mean'] for r in rows]);target=source['targets'][strategy][mode];mean=vals.mean(0)
   summary[mode]={'mean':mean.tolist(),'sd_ddof0':vals.std(0).tolist(),'paper_target':target,'delta':(mean-target).tolist(),'numeric_status':'CLOSE' if np.max(abs(mean-target))<=.015 else 'FAIL','source_prediction_parity':'PASS'}
  report['conditions'][strategy]={'summary':summary,'runs':rows,'distinct_negative_arrays':len({x.tobytes() for x in negative})}
  print(strategy,summary,flush=True)
 report.update(status='PASS',seconds=time.perf_counter()-start,implementation_sha256=sha(P/'relkit/edgebank_l106.py'),runner_sha256=sha(__file__))
 report['artifacts']={x.name:sha(x) for x in out.glob('*.npz')}
 (P/'_analysis_l106_results.json').write_text(json.dumps(report,indent=2)+'\n')
 print('COMPLETE',report['seconds'],flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--raw',type=Path,default=P/'data/l102/wikipedia.csv');a.add_argument('--out',type=Path,default=P/'evidence/l106');args=a.parse_args();run(args.raw,args.out)
