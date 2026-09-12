"""Recompute context scores, paired gaps and dataset-balanced ranks from saved predictions."""
import json,hashlib,itertools
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,friedmanchisquare,studentized_range,t
from sklearn.metrics import accuracy_score,log_loss
ROOT=Path(__file__).resolve().parent

def analyze(data):
 if len(data['config']['datasets'])<3 or len(data['config']['seeds'])<3:raise ValueError('This inferential panel requires at least three datasets and three seeds; inspect smaller-run records descriptively')
 sizes=data['protocol']['context_fractions'];summary=[];matrix=[]
 for dataset in data['config']['datasets']:
  rows=[r for r in data['records'] if r['dataset']==dataset];losses=[]
  for fraction in sizes:
   group=[r for r in rows if r['fraction']==fraction];loss=[];acc=[]
   for r in group:
    p=np.asarray(r['probabilities'],dtype=np.float32);y=np.asarray(r['targets']);score=log_loss(y,p);accuracy=accuracy_score(y,p.argmax(1))
    assert abs(score-r['log_loss'])<1e-7 and accuracy==r['accuracy']
    loss.append(score);acc.append(accuracy)
   losses.append(np.mean(loss));summary.append(dict(dataset=dataset,fraction=fraction,context_rows=len(group[0]['context_ids']),loss_mean=float(np.mean(loss)),loss_sd=float(np.std(loss,ddof=1)) if len(loss)>1 else None,accuracy_mean=float(np.mean(acc)),accuracy_sd=float(np.std(acc,ddof=1)) if len(acc)>1 else None,stage_seconds={k:float(np.mean([r[k] for r in group])) for k in ['preprocessing_seconds','column_seconds','row_seconds','icl_seconds']}))
  matrix.append(losses)
 matrix=np.array(matrix);ranks=rankdata(matrix,axis=1).mean(0);f=friedmanchisquare(*matrix.T)
 permutations=list(itertools.permutations(range(1,len(sizes)+1)))
 null=[np.array(v).mean(0) for v in itertools.product(permutations,repeat=len(matrix))]
 observed=float(np.sum((ranks-(len(sizes)+1)/2)**2))
 exact=sum(float(np.sum((v-(len(sizes)+1)/2)**2))>=observed-1e-12 for v in null)/len(null)
 gaps=[]
 for dataset in data['config']['datasets']:
  values=[]
  for seed in data['config']['seeds']:
   rows=[r for r in data['records'] if r['dataset']==dataset and r['seed']==seed];assert all(r['query_ids']==rows[0]['query_ids'] for r in rows)
   rows=sorted(rows,key=lambda r:r['fraction']);assert all(set(a['context_ids'])<=set(b['context_ids']) for a,b in zip(rows,rows[1:]))
   assert not set(rows[-1]['context_ids'])&set(rows[-1]['query_ids']);assert len(set(rows[-1]['context_ids'])|set(rows[-1]['query_ids']))==rows[-1]['rows']
   values.append(rows[-1]['log_loss']-rows[0]['log_loss'])
  mean=float(np.mean(values));half=float(t.ppf(.975,len(values)-1)*np.std(values,ddof=1)/np.sqrt(len(values)))
  gaps.append(dict(dataset=dataset,full_minus_small_loss=values,mean=mean,conditional_t95=[mean-half,mean+half]))
 return dict(status='PASS',evidence_payload_sha256=hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest(),exact_permutation_p=exact,exact_permutation_draws=len(null),summary=summary,mean_ranks=ranks.tolist(),friedman_statistic=float(f.statistic),friedman_p=float(f.pvalue),nemenyi_cd=float(studentized_range.ppf(.95,len(sizes),np.inf)/np.sqrt(2)*np.sqrt(len(sizes)*(len(sizes)+1)/(6*len(matrix)))),paired_gaps=gaps,total_stage_seconds=sum(r[k] for r in data['records'] for k in ['preprocessing_seconds','column_seconds','row_seconds','icl_seconds']))
if __name__=='__main__':
 result=analyze(json.loads((ROOT/'_verify_l066_v2_results.json').read_text()));(ROOT/'_analysis_l066_v2_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
