"""Independent persisted-prediction and negative-collision audit."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
P=Path(__file__).resolve().parent;E=P/'evidence/l106';r=json.loads((P/'_analysis_l106_results.json').read_text())
for name,digest in r['artifacts'].items():assert hashlib.sha256((E/name).read_bytes()).hexdigest()==digest
# Parse only the endpoint/time projection independently from the raw CSV.
rows=[]
with (P/'data/l102/wikipedia.csv').open() as f:
 next(f)
 for line in f:
  fields=line.split(',',3);rows.append((int(fields[0])+1,int(fields[1])+8228,float(fields[2])))
events=np.array(rows);test=np.load(E/'split.npz')['test_ids'];pos=events[test,:2].astype(np.int64)
base=np.load(E/'predictions-rnd.npz');count=0;gaps={}
for strategy,record in r['conditions'].items():
 pred=np.load(E/f'predictions-{strategy}.npz');neg=np.load(E/f'negatives-{strategy}.npz')['edges'];gaps[strategy]={}
 assert neg.shape==(5,len(test),2)
 for mode in ['unlimited','window']:
  values=pred[mode];assert values.shape==(5,len(test),2)
  np.testing.assert_array_equal(values[:,:,0],base[mode][:,:,0])
  gap=[]
  for k,scores in enumerate(values):
   y=np.r_[np.ones(len(test)),np.zeros(len(test))];s=np.r_[scores[:,0],scores[:,1]]
   pooled=np.array([average_precision_score(y,s),roc_auc_score(y,s)])
   np.testing.assert_allclose(pooled,record['runs'][k][mode]['pooled'],rtol=0,atol=1e-14)
   gap.append((pooled-record['runs'][k][mode]['batch_mean']).tolist());count+=1
  gaps[strategy][mode]=np.mean(gap,axis=0).tolist()
 for k in range(5):
  collisions=0
  for start in range(0,len(test),200):
   p=pos[start:start+200];n=neg[k,start:start+200]
   # Injective encoding for this authenticated node range, independently vectorized.
   collisions+=int(np.isin(n[:,0]*10000+n[:,1],p[:,0]*10000+p[:,1]).sum())
  assert collisions==record['runs'][k]['collisions']
out={'status':'PASS','persisted_prediction_conditions':count,'positive_scores_fixed_across_samplers':'EXACT','independent_collisions':'EXACT','pooled_metric_oracles':'PASS','pooled_minus_batch_mean':gaps}
(P/'_audit_l106_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
