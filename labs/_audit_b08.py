"""Independent keyed prediction and metric audit; does not call trainer/loss."""
import hashlib,json,math,statistics
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;E=P/'evidence/b08'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(root=E):
 protocol=json.loads((root/'course-protocol.json').read_text());report=json.loads((root/'runs/results.json').read_text())
 assert sha(root/'course-protocol.json')==report['protocol_sha256'],'protocol identity'
 assert sha(P/'relkit/limix_b08.py')==protocol['code_sha256'],'model identity'
 for n,h in protocol['data_files'].items():assert sha(P/n)==h,'data identity'
 expected={(s,a) for s in [0,1,2] for a in ['target','feature','combined']};seen=set();rows=[];baselines=[]
 for r in report['records']:
  key=(r['seed'],r['objective']);assert key in expected and key not in seen,'complete unique arm key';seen.add(key)
  path=root/'runs'/r['file'];assert sha(path)==r['prediction_sha256'],'prediction identity'
  pred=np.load(path);data=np.load(P/f'data/b08/test-{r["seed"]}.npz')
  y=pred['y_pred'];x=pred['x_pred'];assert y.shape==data['y'].shape and x.shape==data['x'].shape,'shape'
  assert np.isfinite(x).all() and np.isfinite(y).all(),'finite predictions'
  target_errors=[];feature_errors=[];base_y=[];base_x=[]
  # Complete identities are (seed,episode,batch,row[,column]); no row-only join.
  for episode in range(16):
   for row in range(24,32):
    true=float(data['y'][episode,0,row]);target_errors.append((float(y[episode,0,row])-true)**2)
    base_y.append((float(data['y'][episode,0,:24].astype('float64').mean())-true)**2)
    columns=np.flatnonzero(data['hidden'][episode,0,row]);assert len(columns)==1
    for col in columns:
     true=float(data['x'][episode,0,row,col]);feature_errors.append((float(x[episode,0,row,col])-true)**2)
     base_x.append((float(data['x'][episode,0,:24,col].astype('float64').mean())-true)**2)
  rows.append(dict(seed=r['seed'],objective=r['objective'],target_mse=statistics.mean(target_errors),feature_mse=statistics.mean(feature_errors),target_count=len(target_errors),feature_count=len(feature_errors)))
  if r['objective']=='target':baselines.append(dict(seed=r['seed'],target_mse=statistics.mean(base_y),feature_mse=statistics.mean(base_x)))
 assert seen==expected,'missing arms'
 for seed in range(3):assert len({r['initial_sha256'] for r in report['records'] if r['seed']==seed})==1,'paired initialization'
 summary=[]
 for arm in ['target','feature','combined']:
  s=dict(objective=arm)
  for metric in ['target_mse','feature_mse']:
   vals=[r[metric] for r in rows if r['objective']==arm];s[metric]=statistics.mean(vals);s[metric+'_sd']=statistics.stdev(vals)
  summary.append(s)
 paired=[dict(seed=s,combined_minus_target=next(r['target_mse'] for r in rows if r['seed']==s and r['objective']=='combined')-next(r['target_mse'] for r in rows if r['seed']==s and r['objective']=='target')) for s in range(3)]
 return dict(status='PASS',arms=9,target_predictions=1152,scored_feature_predictions=1152,rows=rows,summary=summary,baselines=baselines,paired=paired,identity='seed/episode/batch/row/column',scope='same synthetic generator; no dataset ranking',learner='PENDING_WRITTEN_DEFENSE')
if __name__=='__main__':
 r=audit();(E/'course-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
