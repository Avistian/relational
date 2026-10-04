"""Independent metric, preprocessing/linear-head reconstruction and corruption checks."""
import hashlib,json,tempfile,shutil
from pathlib import Path
import numpy as np
from _audit_b23 import audit
from _test_b23 import checks
from relkit.comparison_b23 import admit_grid,paired_comparison,claim_status

def rank_auc(y,p):
 order=np.argsort(p,kind='stable');ranks=np.empty(len(p));i=0
 while i<len(p):
  j=i+1
  while j<len(p) and p[order[j]]==p[order[i]]:j+=1
  ranks[order[i:j]]=(i+1+j)/2;i=j
 n=int(y.sum());return float((ranks[y==1].sum()-n*(n+1)/2)/(n*(len(y)-n)))
def verify(e):
 e=Path(e);report=audit(e);assert checks(admit_grid,paired_comparison,claim_status)=='PASS';errors=[];data=np.load(e/'packet/prepared.npz');baseline_errors=[]
 for phase in ['pilot-1','full-1','baseline-1']:
  for row in json.loads((e/phase/'receipt.json').read_text())['records']:
   x=np.load(e/phase/f"{row['arm']}-{row['seed']}.npz");errors.append(abs(rank_auc(x['label'],x['probability'])-row['auc']))
   if row['arm']=='Logistic':
    X=data['X_train'][data['support'][row['seed']]].copy();Q=data['X_test'].copy()
    med=np.array([float(np.median(c[~np.isnan(c)])) if np.any(~np.isnan(c)) else 0. for c in X.T]);X=np.where(np.isnan(X),med,X).astype(data['X_train'].dtype);Q=np.where(np.isnan(Q),med,Q).astype(data['X_test'].dtype)
    X=X.astype(float);mean=X.mean(0);sd=X.std(0);sd[sd==0]=1
    np.testing.assert_allclose(med,x['median']);np.testing.assert_allclose(mean,x['mean']);np.testing.assert_allclose(sd,x['scale'],rtol=1e-10,atol=1e-10)
    Q-=mean.astype(Q.dtype);Q/=sd.astype(Q.dtype);z=Q@x['coef'][0]+x['intercept'][0];expected=1/(1+np.exp(-z));baseline_errors.append(float(np.max(abs(expected-x['probability']))))
 assert max(errors)<1e-12 and max(baseline_errors)<1e-10
 wrong=[(lambda r:dict(runs=40,predictions=28080,paper_runs=30,course_runs=10),paired_comparison,claim_status),(admit_grid,lambda *a:dict(per_seed=[0]*10,positive=0,seeds=list(range(10))),claim_status),(admit_grid,paired_comparison,lambda *a:dict(numerical='CLOSE',historical='ESTABLISHED',deployment_validity='ESTABLISHED',learner='PASS'))]
 for fs in wrong:
  try:checks(*fs)
  except (AssertionError,ValueError):pass
  else:raise AssertionError('Wrong learner accepted')
 rejected=[]
 for mode in ['bytes','missing_run','duplicate_run','keys','supports','labels','probability','baseline_missing']:
  with tempfile.TemporaryDirectory() as td:
   q=Path(td)/'e';shutil.copytree(e,q);path=q/'pilot-1/RDBPFN-0.npz';rp=q/('baseline-1/receipt.json' if mode=='baseline_missing' else 'pilot-1/receipt.json');receipt=json.loads(rp.read_text())
   if mode=='bytes':path.write_bytes(path.read_bytes()+b'corrupt')
   elif mode in ['missing_run','duplicate_run','baseline_missing']:
    if mode=='duplicate_run':receipt['records'].append(receipt['records'][0])
    else:receipt['records'].pop()
    rp.write_text(json.dumps(receipt))
   else:
    with np.load(path) as x:arrays={k:x[k].copy() for k in x.files}
    if mode=='keys':arrays['keys'][0]=arrays['keys'][1]
    if mode=='supports':arrays['support_keys'][0]=arrays['support_keys'][1]
    if mode=='labels':arrays['label'][0]=1-arrays['label'][0]
    if mode=='probability':arrays['probability'][0]=np.nan
    np.savez_compressed(path,**arrays);receipt['records'][0]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();rp.write_text(json.dumps(receipt))
   if mode!='bytes':
    m=json.loads((q/'packet-manifest.json').read_text());m['files']={n:hashlib.sha256((q/n).read_bytes()).hexdigest() for n in m['files']};(q/'packet-manifest.json').write_text(json.dumps(m))
   try:audit(q)
   except (ValueError,AssertionError):rejected.append(mode)
   else:raise AssertionError('Corruption accepted '+mode)
 return dict(status='PASS',independent_rank_auc_runs=len(errors),maximum_auc_error=max(errors),baseline_reconstructed=10,maximum_baseline_probability_error=max(baseline_errors),wrong_learner_functions_rejected=3,corruptions_rejected=rejected,report_parity=report==json.loads((e/'report.json').read_text()))
if __name__=='__main__':
 p=Path(__file__).resolve().parent;r=verify(p/'evidence/b23');assert r['report_parity'];(p/'_verify_b23_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
