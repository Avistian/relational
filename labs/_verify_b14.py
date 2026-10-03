"""Independent course oracle, complete key audit and interventions."""
import copy,json,math,hashlib
from pathlib import Path
import numpy as np
from relkit.flatten_b14 import make_world,flatten,standardize,predict,keyed_mse
P=Path(__file__).resolve().parent;E=P/'evidence/b14'
def audit(report):
 assert report['fits']==12 and len(report['conditions'])==12
 seen=set();count=0
 for r in report['conditions']:
  signature=(r['seed'],r['features'],r['backbone']);assert signature not in seen;seen.add(signature)
  w=make_world(r['seed']);keys=w['query'];rows=r['predictions'];assert len(rows)==len(keys)==256
  lookup={tuple(x['key']):x for x in rows};assert len(lookup)==len(rows) and set(lookup)==set(keys)
  errors=[];rolling=[]
  for k,y in zip(keys,w['target']):
   x=lookup[k];assert x['label']==float(y);assert math.isfinite(x['prediction'])
   errors.append((x['prediction']-y)**2);rolling.append((x['rolling_prediction']-y)**2)
  assert abs(math.fsum(errors)/len(errors)-r['mse'])<1e-12
  assert abs(math.fsum(rolling)/len(rolling)-r['rolling_mse'])<1e-12
  count+=len(rows)
 assert seen=={(s,f,b) for s in range(3) for f in ['entity-only','relational'] for b in ['ridge','rbf']}
 for s in range(3):
  for f in ['entity-only','relational']:
   a=[r for r in report['conditions'] if r['seed']==s and r['features']==f]
   assert len({r['input_sha256'] for r in a})==1
 return count
if __name__=='__main__':
 r=json.loads((E/'diagnostic.json').read_text());count=audit(r);rejected=0
 for kind in ['missing','duplicate','label','score','nan']:
  bad=copy.deepcopy(r);c=bad['conditions'][0]
  if kind=='missing':c['predictions'].pop()
  if kind=='duplicate':c['predictions'][0]=c['predictions'][1]
  if kind=='label':c['predictions'][0]['label']+=1
  if kind=='score':c['mse']+=.01
  if kind=='nan':c['predictions'][0]['prediction']=float('nan')
  try:audit(bad)
  except (AssertionError,ValueError):rejected+=1
  else:raise AssertionError('Corruption accepted: '+kind)
 w=make_world(0);s=flatten(w['entities'],w['events'],w['support'],100);q=flatten(w['entities'],w['events'],w['query'],100)
 # Independent event-filter/reduction, expressed as a streaming oracle.
 expected=[]
 for entity,t in w['query']:
  rows=sorted((e for e in w['events'] if e[1]==entity and max(e[2]+1,e[3])<=min(t,100)),key=lambda e:(e[2],e[0]))
  expected.append([w['entities'][entity],len(rows),math.fsum(e[4] for e in rows)/len(rows),rows[-1][4]])
 np.testing.assert_allclose(q,expected,rtol=0,atol=1e-14)
 a,b=standardize(s,q);before=predict(a,w['y'],b,'rbf');w['target'][:]=1e9
 np.testing.assert_array_equal(before,predict(a,w['y'],b,'rbf'))
 future=w['events']+[(999999,0,105,106,1e6)];np.testing.assert_array_equal(q,flatten(w['entities'],future,w['query'],100))
 assert not np.array_equal(flatten(w['entities'],future,w['query'],120),flatten(w['entities'],w['events'],w['query'],120))
 out=dict(status='PASS',independently_scored_predictions=count,rolling_predictions=count,corruptions_rejected=rejected,paired_input_groups=6,independent_feature_oracle='PASS',future_row_intervention='PASS',hidden_label_intervention='PASS',learner='PENDING_WRITTEN_DEFENSE')
 # Independent linear least-squares / symmetric-eigensolver prediction oracle.
 max_error=0.
 for row in r['conditions']:
  world=make_world(row['seed']);cols=[0] if row['features']=='entity-only' else [0,1,2,3]
  support=flatten(world['entities'],world['events'],world['support'],100)[:,cols]
  query=flatten(world['entities'],world['events'],world['query'],100)[:,cols]
  mu=support.mean(0);scale=support.std(0);scale[scale==0]=1.;x=(support-mu)/scale;z=(query-mu)/scale;y=world['y'];offset=y.mean();yc=y-offset
  if row['backbone']=='ridge':
   coef=np.linalg.lstsq(np.vstack([x,np.eye(x.shape[1])]),np.r_[yc,np.zeros(x.shape[1])],rcond=None)[0];expected=offset+z@coef
  else:
   # Dot-product distances independently replace broadcast differences.
   gram=np.exp(-np.maximum(0,(x*x).sum(1)[:,None]+(x*x).sum(1)[None,:]-2*x@x.T)/len(cols))
   vals,vecs=np.linalg.eigh(gram);weights=vecs@((vecs.T@yc)/(vals+1))
   cross=np.exp(-np.maximum(0,(z*z).sum(1)[:,None]+(x*x).sum(1)[None,:]-2*z@x.T)/len(cols));expected=offset+cross@weights
  actual=np.array([a['prediction'] for a in row['predictions']]);error=float(np.max(np.abs(actual-expected)));assert error<1e-10;max_error=max(max_error,error)
 out['independent_predictor_oracle_max_abs_error']=max_error
 (P/'_verify_b14_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
