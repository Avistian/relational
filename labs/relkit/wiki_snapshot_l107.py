"""Complete Wikipedia course comparison. This is not a paper-table experiment."""
# %% Course imports
import copy,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
try:
 from snapshot_l107 import SnapshotGRU,normalized_adjacency
 from tgn_l102 import TGN,load_wikipedia,temporal_neighbors,NegativeSampler
except ImportError:
 from relkit.snapshot_l107 import SnapshotGRU,normalized_adjacency
 from relkit.tgn_l102 import TGN,load_wikipedia,temporal_neighbors,NegativeSampler

# %% Frozen questions and candidate schedule shared across architectures

def wiki_data(directory,seed=0):
 nodes,edges,d,audit=load_wikipedia(directory)
 # Retain exactly the released training/val/test populations. No extra held-out past ingested.
 events={key:np.concatenate([d[s][key] for s in ['train','val','test']]) for key in ['u','v','t','e']}
 events['split']=np.concatenate([np.full(len(d[s]['u']),i) for i,s in enumerate(['train','val','test'])])
 # Same candidates in every arm; frozen across epochs. This differs from the TGN paper sampler.
 events['negative']=np.concatenate([NegativeSampler(d['train'] if i==0 else d['full'],100+seed if i==0 else i).sample(len(d[s]['u'])) for i,s in enumerate(['train','val','test'])])
 return nodes,edges,events,audit

def pooled_metrics(y,p):
 return {'ap':float(average_precision_score(y,p)),'auc':float(roc_auc_score(y,p))}

# %% Completed-window snapshot representation from raw edge features

def snapshot_inputs(events,edges,n,width):
 bins=np.floor(events['t']/width).astype(int);last=int(bins.max());out=[]
 for k in range(last+1):
  at=np.flatnonzero(bins==k);u=events['u'][at];v=events['v'][at]
  # Weighted interaction graph; repeat counts remain in A. No current-bin features in its forecasts.
  pairs=np.stack([np.r_[u,v],np.r_[v,u]],1)
  adj=normalized_adjacency(pairs,np.ones(len(pairs)),n)
  x=np.zeros((n,edges.shape[1]+2),np.float32);count=np.bincount(np.r_[u,v],minlength=n)
  np.add.at(x[:,:-2],u,edges[events['e'][at]]);np.add.at(x[:,:-2],v,edges[events['e'][at]])
  x[:,:-2]/=np.maximum(count,1)[:,None];x[:,-2]=np.log1p(count);x[:,-1]=count>0
  out.append((adj,torch.as_tensor(x),at))
 return out

# %% Full-data snapshot training and validation-only selection

def train_snapshot(nodes,edges,events,width,seed,output,device='cpu',epochs=10,pilot=False):
 output=Path(output);output.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed)
 model=SnapshotGRU(edges.shape[1]+2,32).to(device);opt=torch.optim.Adam(model.parameters(),lr=.001)
 windows=snapshot_inputs(events,edges,len(nodes),width);trace=[];best=-float('inf');selected=None;start=time.perf_counter()
 def epoch_pass(training=False,split=None):
  state=torch.zeros(len(nodes),32,device=device);ys=[];ps=[];ids=[];losses=[];previous=None
  model.train(training)
  with torch.set_grad_enabled(training):
   for k,(a,x,at) in enumerate(windows):
    if training and len(at) and not np.any(events['split'][at]==0):break
    if previous is not None:
     aa,xx=previous;state=model.step(aa.to(device),xx.to(device),state.detach())
    idx=at[events['split'][at]==(0 if training else split)] if split is not None or training else at
    if len(idx):
     p=np.stack([np.r_[events['u'][idx],events['u'][idx]],np.r_[events['v'][idx],events['negative'][idx]]],1)
     y=np.r_[np.ones(len(idx)),np.zeros(len(idx))].astype(np.int64)
     logits=model.score(state,torch.as_tensor(p,device=device));loss=torch.nn.functional.cross_entropy(logits,torch.as_tensor(y,device=device))
     if training:
      opt.zero_grad();loss.backward();opt.step();state=state.detach();losses.append(float(loss.detach()))
     else:ys.append(y);ps.append(logits.softmax(1)[:,1].cpu().numpy());ids.append(np.r_[events['e'][idx],events['e'][idx]])
    previous=(a,x)
    if pilot and k>=8:break
  return (float(np.mean(losses)) if losses else None) if training else (np.concatenate(ys),np.concatenate(ps),np.concatenate(ids))
 for epoch in range(epochs):
  tick=time.perf_counter();loss=epoch_pass(True)
  if pilot:trace.append({'epoch':epoch,'loss':loss,'seconds':time.perf_counter()-tick});break
  y,p,_=epoch_pass(False,1);val=pooled_metrics(y,p)
  if val['ap']>best:best=val['ap'];selected=epoch;torch.save(model.state_dict(),output/'best.pt')
  row={'epoch':epoch,'loss':loss,'val':val,'seconds':time.perf_counter()-tick};trace.append(row);print(json.dumps({'arm':f'snapshot-{width}','seed':seed,**row}),flush=True)
  (output/'trace.json').write_text(json.dumps(trace,indent=2))
 result={'status':'PILOT' if pilot else 'COMPLETE','arm':f'snapshot-{width}','seed':seed,'epochs':len(trace),'selected_epoch':selected,'trace':trace,'seconds':time.perf_counter()-start}
 if not pilot:
  model.load_state_dict(torch.load(output/'best.pt',weights_only=True));y,p,ids=epoch_pass(False,2)
  result['test']=pooled_metrics(y,p);np.savez_compressed(output/'predictions.npz',y=y,prob=p,edge=ids)
 (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result

# %% Strict-past TGN batches do not split equal timestamps

def time_batches(events,which,size=200):
 idx=np.flatnonzero(events['split']==which);start=0
 while start<len(idx):
  end=min(start+size,len(idx))
  while end<len(idx) and events['t'][idx[end]]==events['t'][idx[end-1]]:end+=1
  yield idx[start:end];start=end

# %% Full-data compact TGN course trainer, same candidates and selection budget

def train_tgn_course(nodes,edges,events,seed,output,device='cpu',epochs=10,pilot=False):
 output=Path(output);output.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed);np.random.seed(seed)
 model=TGN(np.zeros((len(nodes),32),np.float32),edges,dropout=0.,neighbors=10).to(device)
 model.finder=temporal_neighbors(events['u'],events['v'],events['t'],events['e'],len(nodes))
 opt=torch.optim.Adam(model.parameters(),lr=.001);trace=[];best=-float('inf');selected=None;start=time.perf_counter()
 def run(which,training=False):
  ys=[];ps=[];ids=[];losses=[];model.train(training)
  with torch.set_grad_enabled(training):
   for batch,idx in enumerate(time_batches(events,which)):
    p,q=model.probabilities(events['u'][idx],events['v'][idx],events['negative'][idx],events['t'][idx],events['e'][idx])
    loss=(torch.nn.functional.binary_cross_entropy(p,torch.ones_like(p))+torch.nn.functional.binary_cross_entropy(q,torch.zeros_like(q)))/2
    if training:opt.zero_grad();loss.backward();opt.step();model.detach_state();losses.append(float(loss.detach()))
    else:ys.append(np.r_[np.ones(len(idx)),np.zeros(len(idx))]);ps.append(np.r_[p.cpu().numpy(),q.cpu().numpy()]);ids.append(np.r_[events['e'][idx],events['e'][idx]])
    if pilot and batch>=4:break
  return float(np.mean(losses)) if training else (np.concatenate(ys),np.concatenate(ps),np.concatenate(ids))
 for epoch in range(epochs):
  tick=time.perf_counter();model.reset_state();loss=run(0,True)
  if pilot:trace.append({'epoch':epoch,'loss':loss,'seconds':time.perf_counter()-tick});break
  y,p,_=run(1);val=pooled_metrics(y,p)
  if val['ap']>best:
   best=val['ap'];selected=epoch
   torch.save({'weights':{k:v for k,v in model.state_dict().items() if k not in ['node_features','edge_features']},'state':model.snapshot()},output/'best.pt')
  row={'epoch':epoch,'loss':loss,'val':val,'seconds':time.perf_counter()-tick};trace.append(row);print(json.dumps({'arm':'tgn','seed':seed,**row}),flush=True)
  (output/'trace.json').write_text(json.dumps(trace,indent=2))
 result={'status':'PILOT' if pilot else 'COMPLETE','arm':'tgn','seed':seed,'epochs':len(trace),'selected_epoch':selected,'trace':trace,'seconds':time.perf_counter()-start}
 if not pilot:
  saved=torch.load(output/'best.pt',map_location=device,weights_only=False);model.load_state_dict(saved['weights'],strict=False);model.restore(saved['state']);y,p,ids=run(2)
  result['test']=pooled_metrics(y,p);np.savez_compressed(output/'predictions.npz',y=y,prob=p,edge=ids)
 (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
