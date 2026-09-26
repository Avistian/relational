"""Replay downloaded fresh checkpoints in the local CPU runtime; no training or cloud calls."""
import argparse,json,time,hashlib,platform,importlib.metadata
from pathlib import Path
import numpy as np
import torch
from relkit.checkpoint_l110 import TGN,load_wikipedia,temporal_neighbors,restore_checkpoint,evaluate,NegativeSampler
P=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--seeds',default='0,1,2,3,4,5,6,7,8,9');p.add_argument('--arms',default='release,clean');p.add_argument('--available',action='store_true');a=p.parse_args();torch.set_num_threads(1)
nodes,edges,data,_=load_wikipedia(P/'data/l102');finder=temporal_neighbors(data['full']['u'],data['full']['v'],data['full']['t'],data['full']['e'],len(nodes));results=[];start=time.perf_counter()
for seed in map(int,a.seeds.split(',')):
 for arm in a.arms.split(','):
  root=P/'results/l110/checkpoints'/f'seed-{seed}'/arm;path=root/f'seed-{seed}.pt';
  if not path.exists():
   if a.available:continue
   raise FileNotFoundError(path)
  report=P/'evidence/l110'/f'seed-{seed}'/arm/'replay-cpu.json'
  identity={'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256((P/'relkit/checkpoint_l110.py').read_bytes()).hexdigest(),'replay_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'predictions_sha256':hashlib.sha256((report.parent/f'seed-{seed}-predictions.npz').read_bytes()).hexdigest(),'runtime':{k:importlib.metadata.version(k) for k in ['torch','numpy','pandas','scikit-learn']},'python':platform.python_version()}
  if report.exists():
   cached=json.loads(report.read_text())
   if cached['identity']==identity:
    results.extend(cached['results']);print('Reused authenticated replay',seed,arm,flush=True);continue
  local=[]
  saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['result']['seed']==seed and saved['result']['arm']==arm
  model=TGN(nodes,edges);model.finder=finder;z=np.load(P/'evidence/l110'/f'seed-{seed}'/arm/f'seed-{seed}-predictions.npz')
  for lane,key,rs in [('all','test',2),('new','new_test',3)]:
   restore_checkpoint(model,saved);pool=data['full'] if lane=='all' else data['new_test'];r,preds=evaluate(model,data[key],NegativeSampler(pool,rs),clean=arm=='clean')
   err=0.
   for field in ['edges','negative','batch_id','positive','negative_score']:
    arr=np.concatenate([x[field] for x in preds]);ref=z[lane+'_'+field]
    if field in ['positive','negative_score']:err=max(err,float(np.max(np.abs(arr-ref))));assert np.allclose(arr,ref,atol=2e-5,rtol=0),(seed,arm,lane,err)
    else:np.testing.assert_array_equal(arr,ref)
   assert abs(r['ap']-saved['result'][key]['ap'])<1e-5
   row={'seed':seed,'arm':arm,'population':lane,'events':len(data[key]['e']),'max_probability_error':err,'ap_error':abs(r['ap']-saved['result'][key]['ap'])};results.append(row);local.append(row);print(row,flush=True)
  report.write_text(json.dumps({'status':'PASS','identity':identity,'results':local},indent=2))
  del model,saved
out={'status':'PASS' if len(results)==40 else 'PARTIAL','seconds':time.perf_counter()-start,'results':results,'scope':'Fresh GPU artifacts replayed with visible model in distinct local CPU runtime; no retraining','source_model_replay':'Separate original-source GPU replay for release seed0'}
(P/'_replay_l110_results.json').write_text(json.dumps(out,indent=2))
