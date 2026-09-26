"""Full frozen-checkpoint evaluation; source replay and inference interventions."""
import argparse,hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np
import torch
from relkit.tgat_l103 import load_wikipedia,TGAT,NeighborFinder
from relkit.leakage_l104 import replay_release,records_from_archive,batch_ap,paired_ap
from relkit.sampling_l108 import TemporalIndex,expansion_size
P=Path(__file__).resolve().parent
FILES=['relkit/tgat_l103.py','relkit/leakage_l104.py','relkit/sampling_l108.py','_run_l108.py','_inputs_l108.json']
ARMS={'uniform20':('uniform',20,np.inf),'uniform5':('uniform',5,np.inf),'recent20':('recent',20,np.inf),'day20':('uniform',20,86400.)}
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def fingerprint():return hashlib.sha256(''.join(sha(P/f) for f in FILES).encode()).hexdigest()
def synchronize(device):
 if str(device).startswith('cuda'):torch.cuda.synchronize()

@torch.no_grad()
def score(model,events,questions,fanout,device,deadline):
 model.eval();position=np.searchsorted(events['e'],questions['e'])
 np.testing.assert_array_equal(events['e'][position],questions['e'])
 out={k:questions[k].copy() for k in ['e','negative','batch']};out.update(p=np.empty(len(position),np.float32),n=np.empty(len(position),np.float32))
 synchronize(device);start=time.perf_counter()
 for b in np.unique(questions['batch']):
  if time.monotonic()>deadline:raise TimeoutError('Incomplete evaluation: runtime cap')
  mask=questions['batch']==b;idx=position[mask]
  p,n=model.contrast(events['u'][idx],events['v'][idx],questions['negative'][mask],events['t'][idx],fanout)
  out['p'][mask],out['n'][mask]=p.cpu().numpy(),n.cpu().numpy()
 synchronize(device)
 return out,time.perf_counter()-start

def run(seed,checkpoint_root,data_dir,output,device='cpu',pilot=False,max_seconds=2850):
 started=time.monotonic();deadline=started+max_seconds;dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((P/'_inputs_l108.json').read_text());src=Path(checkpoint_root)/f'seed-{seed}'
 assert sha(P/'relkit/tgat_l103.py')==manifest['implementation_sha256']
 for name,h in manifest['seeds'][str(seed)].items():assert sha(src/name)==h,(seed,name)
 n,e,d,a=load_wikipedia(data_dir);assert a['split_sha256']==manifest['split_sha256']
 with np.load(Path(data_dir)/'processed.npz') as data:
  for key,item in manifest['processed_arrays'].items():assert hashlib.sha256(data[key].tobytes()).hexdigest()==item['sha256']
 identity={'fingerprint':fingerprint(),'seed':seed,'pilot':pilot,'device':device,'runtime':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform()},'checkpoint':manifest['seeds'][str(seed)]['selected.pt'],'source_commit':manifest['source_commit'],'training':'REUSED_L103','arms':{k:[p,f,'all' if np.isinf(w) else w] for k,(p,f,w) in ARMS.items()}}
 if (dest/'result.json').exists():
  previous=json.loads((dest/'result.json').read_text());assert previous['identity']==identity,'Resume identity mismatch'
  assert previous['predictions_sha256']==sha(dest/'predictions.npz'),'Artifact mismatch'
  return previous
 checkpoint=torch.load(src/'selected.pt',map_location=device,weights_only=False)
 model=TGAT(NeighborFinder(d['full'],len(n),release=True),n,e).to(device)
 loaded=model.load_state_dict(checkpoint['weights'],strict=False)
 assert set(loaded.missing_keys)=={'n_feat_th','e_feat_th','edge_raw_embed.weight','node_raw_embed.weight'} and not loaded.unexpected_keys
 archive=np.load(src/'predictions.npz');saved={};release=None
 if not pilot:
  release,outputs=replay_release(model,d,checkpoint,archive,deadline)
  for lane,records in outputs.items():
   for key,value in records.items():saved[f'release_{lane}_{key}']=value
  print(json.dumps({'seed':seed,'stage':'release replay','seconds':time.monotonic()-started}),flush=True)
 index=TemporalIndex(d['full'],len(n));model.ngh_finder=index
 # Warm model/CUDA kernels once; warmup never enters measured evidence.
 q=records_from_archive(archive,'all');small={k:v[:30] for k,v in q.items()}
 score(model,d['test'],small,20,device,deadline)
 metrics={};base={}
 # Rotate order by seed to avoid always measuring one arm with the coldest cache.
 names=list(ARMS);names=names[seed%4:]+names[:seed%4]
 for arm in names:
  policy,fanout,window=ARMS[arm];metrics[arm]={};index.policy=policy;index.window=window
  for lane,split in [('all','test'),('new','new_test')]:
   questions=records_from_archive(archive,lane)
   if pilot:questions={k:v[:120] for k,v in questions.items()}
   np.random.seed(108+seed);index.audit={k:0 for k in index.audit}
   if device=='cuda':torch.cuda.reset_peak_memory_stats()
   pred,seconds=score(model,d[split],questions,fanout,device,deadline)
   assert index.audit['nonpast_records']==index.audit['expired_records']==0
   metrics[arm][lane]={'seconds':seconds,'events':len(pred['e']),'events_per_second':len(pred['e'])/seconds,'batch_ap':batch_ap(pred),'audit':index.audit.copy(),'tree_nodes_per_root':expansion_size(2,fanout),'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None}
   base[arm,lane]=pred
   for key,value in pred.items():saved[f'{arm}_{lane}_{key}']=value
  print(json.dumps({'seed':seed,'stage':arm,'seconds':time.monotonic()-started}),flush=True)
 for arm in ARMS:
  for lane in ['all','new']:metrics[arm][lane].update(paired_ap(base['uniform20',lane],base[arm,lane]))
 np.savez_compressed(dest/'predictions.npz',**saved)
 result={'status':'PILOT' if pilot else 'COMPLETE','identity':identity,'release':release,'metrics':metrics,'elapsed_seconds':time.monotonic()-started,'predictions_sha256':sha(dest/'predictions.npz'),'index_array_bytes':index.array_bytes,'full_paper':'NOT_ESTABLISHED','historical_identity':'INCOMPARABLE','interventions':'FIXED_WEIGHT_COURSE_EXTENSION','checkpoint_training':'NOT_RUN_IN_L108'}
 (dest/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--checkpoint-root',default=str(P/'results/l103/gpu'));p.add_argument('--data-dir',default=str(P/'l103-cache'));p.add_argument('--output',default=str(P/'results/l108'));p.add_argument('--device',default='cpu');p.add_argument('--pilot',action='store_true');p.add_argument('--max-seconds',type=int,default=2850);a=p.parse_args();torch.set_num_threads(1)
 r=run(a.seed,a.checkpoint_root,a.data_dir,Path(a.output)/f'seed-{a.seed}',a.device,a.pilot,a.max_seconds);print(r['status'])
