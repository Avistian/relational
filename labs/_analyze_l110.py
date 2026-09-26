"""Independently reconstruct AP from prediction arrays and verify complete paired identities."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from relkit.checkpoint_l110 import load_wikipedia
P=Path(__file__).resolve().parent;O=P/'evidence/l110';src=json.loads((P/'_sources_l110.json').read_text());_,_,data,audit=load_wikipedia(P/'data/l102')
def independent_ap(y,p):
 order=np.argsort(-p,kind='stable');y=y[order];p=p[order];last=np.r_[np.flatnonzero(p[:-1]!=p[1:]),len(p)-1];tp=np.cumsum(y)[last];rec=tp/y.sum();prec=tp/(last+1)
 return float(np.sum(np.diff(np.r_[0.,rec])*prec))
# Exercise exact ties, shuffled order and a one-positive case against the library definition.
for y,p in [(np.array([1,0,1,0]),np.array([.5,.5,.2,.1])),(np.array([0,1,0]),np.array([.9,.1,.1]))]:assert abs(independent_ap(y,p)-average_precision_score(y,p))<1e-14
pilot_identity=json.loads((O/'pilot/identity.json').read_text())
paired=[];maxerr=0.;count=0;cost=0.;epochs={k:0 for k in ['release','clean']};filehash={}
for seed in range(10):
 root=O/f'seed-{seed}';ident=json.loads((root/'identity.json').read_text());assert ident['source_sha256']==src['implementation_sha256'] and ident['seed']==seed and ident['epochs']==50 and ident['preset']=='paper';assert ident['audit']==audit
 for key in ['torch','numpy','python','device']:assert ident[key]==pilot_identity[key],(seed,key)
 assert ident['torch'].startswith('2.8.0') and ident['numpy']=='2.2.6'
 complete=json.loads((root/'completed.json').read_text());assert complete['status']=='COMPLETE';cost+=complete['resource_cost_usd'];record={'seed':seed}
 for arm in ['release','clean']:
  run=json.loads((root/arm/f'seed-{seed}.json').read_text());z=np.load(root/arm/f'seed-{seed}-predictions.npz');assert run['seed']==seed and run['arm']==arm;epochs[arm]+=run['epochs_completed'];record[arm]={}
  trace=run['trace'];best=-float('inf');best_epoch=None;rounds=0
  for i,row in enumerate(trace):
   assert row['epoch']==i
   if best_epoch is None or (row['val_ap']-best)/abs(best)>1e-10:best=row['val_ap'];best_epoch=i;rounds=0
   else:rounds+=1
  assert run['selected_epoch']==(best_epoch if arm=='clean' or run['early_stopped'] else len(trace)-1)
  assert len(trace)==run['epochs_completed'] and (rounds>=5 if run['early_stopped'] else len(trace)==50)
  for lane,key,rngseed in [('all','test',2),('new','new_test',3)]:
   ev=data[key];ids=z[lane+'_edges'];batch=z[lane+'_batch_id'];pos=z[lane+'_positive'];neg=z[lane+'_negative_score'];candidates=z[lane+'_negative'];np.testing.assert_array_equal(ids,ev['e']);assert np.isfinite(pos).all() and np.isfinite(neg).all();assert ((pos>=0)&(pos<=1)).all() and ((neg>=0)&(neg<=1)).all()
   pool=data['full'] if lane=='all' else data['new_test'];sources=np.unique(pool['u']);destinations=np.unique(pool['v']);rng=np.random.RandomState(rngseed);aps=[];aucs=[];last=0
   np.testing.assert_array_equal(np.unique(batch),np.arange(batch.max()+1))
   for j in np.unique(batch):
    ix=np.flatnonzero(batch==j);assert ix[0]==last;last=ix[-1]+1
    assert np.array_equal(ix,np.arange(ix[0],last));size=len(ix)
    rng.randint(0,len(sources),size);expected=destinations[rng.randint(0,len(destinations),size)];np.testing.assert_array_equal(expected,candidates[ix])
    if arm=='clean' and ix[0]>0:assert ev['t'][ix[0]-1]<ev['t'][ix[0]]
    if arm=='release':assert size==min(200,len(ids)-ix[0])
    else:
     nominal=min(ix[0]+200,len(ids));end=int(np.searchsorted(ev['t'],ev['t'][nominal-1],side='right'));assert last==end
    y=np.r_[np.ones(size),np.zeros(size)];p=np.r_[pos[ix],neg[ix]];aps.append(independent_ap(y,p));aucs.append(float(roc_auc_score(y,p)))
   assert last==len(ids);ap=float(np.mean(aps));auc=float(np.mean(aucs));err=abs(ap-run[key]['ap']);maxerr=max(maxerr,err);assert err<1e-12;np.testing.assert_allclose(aps,run[key]['batch_ap'],atol=1e-12,rtol=0);assert abs(auc-run[key]['auc'])<1e-12
   pooled=independent_ap(np.r_[np.ones(len(pos)),np.zeros(len(neg))],np.r_[pos,neg]);count+=len(ids)
   record[arm][lane]={'batch_ap_percent':ap*100,'pooled_ap_percent':pooled*100,'auc':auc,'events':len(ids),'batches':len(aps)}
  record[arm]['selected_epoch']=run['selected_epoch'];record[arm]['epochs_completed']=len(trace)
 paired.append(record)
summary={};delta={}
for arm in ['release','clean']:
 summary[arm]={}
 for lane,target in [('all',98.46),('new',97.81)]:
  a=np.array([r[arm][lane]['batch_ap_percent'] for r in paired]);pooled=np.array([r[arm][lane]['pooled_ap_percent'] for r in paired])
  summary[arm][lane]={'mean_ap_percent':float(a.mean()),'sample_sd_pp':float(a.std(ddof=1)),'pooled_mean_ap_percent':float(pooled.mean()),'pooled_sample_sd_pp':float(pooled.std(ddof=1)),'paper_target':target if arm=='release' else None,'gap_pp':float(a.mean()-target) if arm=='release' else None,'numerical_verdict':('CLOSE' if abs(a.mean()-target)<=.5 else 'OUTSIDE_TOLERANCE') if arm=='release' else 'COURSE_INTERVENTION'}
for lane in ['all','new']:
 a=np.array([r['clean'][lane]['batch_ap_percent']-r['release'][lane]['batch_ap_percent'] for r in paired]);b=np.array([r['clean'][lane]['pooled_ap_percent']-r['release'][lane]['pooled_ap_percent'] for r in paired]);delta[lane]={'mean_batch_delta_pp':float(a.mean()),'sd_batch_delta_pp':float(a.std(ddof=1)),'mean_pooled_delta_pp':float(b.mean()),'sd_pooled_delta_pp':float(b.std(ddof=1))}
pilot=json.loads((O/'pilot/completed.json').read_text());cost+=pilot['resource_cost_usd']
for p in O.rglob('*'):
 if p.is_file() and p.name not in ['manifest.json','summary.json']:filehash[str(p.relative_to(O))]=hashlib.sha256(p.read_bytes()).hexdigest()
r={'status':'COMPLETE','paired':paired,'summary':summary,'paired_deltas':delta,'event_evaluations':count,'max_independent_ap_error':maxerr,'epochs':epochs,'completed_call_resource_usd':cost,'billing_scope':'Function resource estimate including pilot; startup/build/storage not itemized','source_sha256':src['implementation_sha256'],'artifact_hashes':filehash,'historical_identity':'INCOMPARABLE','full_paper_reproduction':'NOT_ESTABLISHED','candidate_rng_reconstruction':'EXACT','validation_selection_reconstruction':'PASS'}
(O/'summary.json').write_text(json.dumps(r,indent=2));(P/'_analysis_l110_results.json').write_text(json.dumps({k:v for k,v in r.items() if k not in ['paired','artifact_hashes']},indent=2));print(json.dumps(r['summary'],indent=2));print('resource USD',cost)
