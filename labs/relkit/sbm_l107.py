"""Complete released SBM protocol, with bounded-memory pair-head backpropagation.
See l107-reproduction.md for deviations from historical runtime and worker RNG.
"""
# %% SBM imports and authenticated data
import hashlib,json,time,copy,urllib.request,tarfile
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import average_precision_score
try:
 from snapshot_l107 import EvolveGCN,PairClassifier,normalized_adjacency,weighted_cross_entropy
except ImportError:
 from relkit.snapshot_l107 import EvolveGCN,PairClassifier,normalized_adjacency,weighted_cross_entropy
SBM_SHA='bfae7d9d89400b42c2ef513eaeef56dbbcfae2745c59f40fb3746f9cf98c8c45'
COMMIT='3f4996ac2a742a69fe6ce6e378b6317518bd99bf'

def load_sbm(directory):
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);raw=directory/'sbm.csv'
 if not raw.exists():
  archive=directory/'sbm.tar.gz';urllib.request.urlretrieve(f'https://raw.githubusercontent.com/IBM/EvolveGCN/{COMMIT}/data/sbm_50t_1000n_adj.csv.tar.gz',archive)
  with tarfile.open(archive) as tar:
   m=next(x for x in tar.getmembers() if x.name.endswith('.csv'));raw.write_bytes(tar.extractfile(m).read())
 assert hashlib.file_digest(raw.open('rb'),'sha256').hexdigest()==SBM_SHA
 edges=pd.read_csv(raw).to_numpy(dtype=np.int64);edges[:,3]-=edges[:,3].min()
 n=int(edges[:,:2].max()+1);last=int(edges[:,3].max());pairs=[];weights=[];degrees=[]
 for t in range(last+1):
  e=edges[edges[:,3]==t];p,w=np.unique(e[:,:2],axis=0,return_counts=True)
  pairs.append(p);weights.append(w);degrees.append(np.bincount(p[:,0],minlength=n))
 f=int(np.max(degrees[:-1]))+1 # source excludes final snapshot from schema scan
 features=[torch.nn.functional.one_hot(torch.as_tensor(d),f).float() for d in degrees]
 adj=[normalized_adjacency(p,w,n) for p,w in zip(pairs,weights)]
 masks=[]
 for p in pairs:
  mask=torch.full((n,1),-float('inf'));mask[np.unique(p)]=0;masks.append(mask)
 return {'n':n,'f':f,'last':last,'pairs':pairs,'adj':adj,'features':features,'masks':masks,'rows':len(edges),'degrees':degrees}

# %% Released smart negative sampler, preserving first-occurrence order

def smart_negatives(positive,existing,n,rng,mult=50):
 number=min(len(positive)*mult,len(positive)*(len(positive)-1)-len(positive))
 # Source draws all source IDs first, then all destinations. Preserve that RNG contract.
 source=rng.choice(positive[:,0],size=number*4,replace=True)
 target=rng.choice(existing,size=number*4,replace=True)
 ids=source*n+target
 first=np.full(n*n,len(ids),dtype=np.int64)
 np.minimum.at(first,ids,np.arange(len(ids)))
 first[positive[:,0]*n+positive[:,1]]=len(ids);first[np.arange(n)*(n+1)]=len(ids)
 accepted=np.flatnonzero(first<len(ids));order=np.argsort(first[accepted]);accepted=accepted[order[:number]]
 return np.stack([accepted//n,accepted%n],axis=1)

def sbm_candidates(data,t,rng=None):
 positive=data['pairs'][t+1];n=data['n']
 if rng is None:
  # Original meshgrid flatten order: destination outer, source inner.
  ids=np.arange(n*n);pair=np.stack([ids%n,ids//n],1)
  truth=np.zeros(n*n,bool);truth[positive[:,0]*n+positive[:,1]]=True
  neg=pair[~truth[pair[:,0]*n+pair[:,1]]]
 else:
  existing=np.concatenate([np.unique(data['pairs'][i]) for i in range(t-5,t+1)])
  neg=smart_negatives(positive,existing,n,rng)
 return np.concatenate([positive,neg]),np.concatenate([np.ones(len(positive)),np.zeros(len(neg))]).astype(np.int64)

# %% Released MAP and average-over-positive-ranks MRR

def sbm_metrics(pairs,labels,probs,n):
 ap=float(average_precision_score(labels,probs))
 score=np.zeros((n,n),dtype=probs.dtype);truth=np.zeros((n,n),dtype=bool)
 score[pairs[:,0],pairs[:,1]]=probs;truth[pairs[:,0],pairs[:,1]]=labels.astype(bool)
 # Match source np.flip(row.argsort()), including its deterministic tie order.
 order=np.flip(score.argsort(axis=1),axis=1)
 ranked=np.take_along_axis(truth,order,axis=1)
 counts=truth.sum(1);rr=(ranked/np.arange(1,n+1)).sum(1)
 mrr=float(np.mean(rr[counts>0]/counts[counts>0]))
 return {'map':ap,'mrr':mrr}

# %% One complete graph-window training step, exact loss normalization

def pair_pass(model,head,data,t,pairs,labels,device,training,weights,chunk=16384):
 a=[x.to(device) for x in data['adj'][t-5:t+1]];x=[v.to(device) for v in data['features'][t-5:t+1]];mask=[v.to(device) for v in data['masks'][t-5:t+1]]
 with torch.set_grad_enabled(training):
  z=model(a,x,mask)
  # Head chunks accumulate dL/dz before a single GCN backward; no truncated history.
  leaf=z.detach().requires_grad_(training);out=[];loss_total=0.
  for start in range(0,len(pairs),chunk):
   p=torch.as_tensor(pairs[start:start+chunk],device=device);y=torch.as_tensor(labels[start:start+chunk],device=device)
   logits=head(leaf,p);loss=weighted_cross_entropy(logits,y,weights)*len(y)/len(labels)
   if training:loss.backward()
   loss_total+=float(loss.detach());out.append(logits.softmax(1)[:,1].detach().cpu().numpy())
  if training:z.backward(leaf.grad)
 return np.concatenate(out),loss_total

# %% Complete released training budget and validation-only checkpoint selection

def train_sbm(data,variant,output,device='cpu',epochs=100,max_seconds=10000,pilot=False):
 output=Path(output);output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 torch.manual_seed(1234);np.random.seed(1234)
 # Source loader allocates random node features even though degree features replace them.
 torch.rand(data['n'],3)
 h=50 if variant=='H' else 51;cls=100 if variant=='H' else 565;lr=.01 if variant=='H' else .005
 model=EvolveGCN(data['f'],h,variant).to(device);head=PairClassifier(h,cls).to(device)
 opt=torch.optim.Adam(model.parameters(),lr=lr);headopt=torch.optim.Adam(head.parameters(),lr=lr)
 weights=torch.tensor([.1,.9] if variant=='H' else [.15,.85],device=device)
 train=list(range(5,int(np.floor(data['last']*.7))));valid=list(range(train[-1]+1,int(np.floor(data['last']*.8))));test=list(range(valid[-1]+1,data['last']))
 rng=np.random.RandomState(1234);trace=[];best=-float('inf');best_state=None;best_epoch=None;stale=0;status='COMPLETE'
 eval_sets={t:sbm_candidates(data,t) for t in valid+test}
 def evaluate(times,save=False):
  rows=[]
  for t in times:
   pairs,y=eval_sets[t]
   if save:torch.save({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if device.startswith('cuda') else []},output/f'test-{t}-rng.pt')
   prob,_=pair_pass(model,head,data,t,pairs,y,device,False,weights)
   metric=sbm_metrics(pairs,y,prob,data['n']);rows.append({'t':t,'count':len(y),**metric})
   if save:np.savez_compressed(output/f'test-{t}.npz',pairs=pairs.astype(np.int32),y=y.astype(np.uint8),prob=prob)
  return {key:float(np.average([r[key] for r in rows],weights=[r['count'] for r in rows])) for key in ['map','mrr']},rows
 for epoch in range(epochs):
  tick=time.perf_counter();losses=[];counts=[]
  for t in train[:1] if pilot else train:
   if time.perf_counter()-start>max_seconds:status='INCOMPLETE';break
   pairs,y=sbm_candidates(data,t,rng);opt.zero_grad();headopt.zero_grad()
   _,loss=pair_pass(model,head,data,t,pairs,y,device,True,weights);opt.step();headopt.step();losses.append(loss);counts.append(len(y))
  if status=='INCOMPLETE':break
  row={'epoch':epoch,'train_loss':float(np.mean(losses)),'train_candidates':counts,'seconds_train':time.perf_counter()-tick}
  if epoch>5 or pilot:
   metric,_=evaluate(valid[:1] if pilot else valid);row['valid']=metric
   if metric['map']>best:
    best=metric['map'];best_epoch=epoch;stale=0;best_state={'model':copy.deepcopy(model.state_dict()),'head':copy.deepcopy(head.state_dict())}
    # Preserve stochastic RReLU draws at validation-selected test epochs.
    result,testrows=evaluate(test[:1] if pilot else test,save=True);row['test_at_improvement']=result
    torch.save(best_state,output/'best.pt');(output/'selected.json').write_text(json.dumps({'epoch':epoch,'valid':metric,'test':result,'per_snapshot':testrows},indent=2))
   else:stale+=1
  row['seconds']=time.perf_counter()-tick;trace.append(row)
  (output/'trace.json').write_text(json.dumps(trace,indent=2));print(json.dumps({'variant':variant,**row}),flush=True)
  if stale>50:break
 summary={'status':'PILOT' if pilot else status,'variant':variant,'epochs':len(trace),'selected_epoch':best_epoch,'seconds':time.perf_counter()-start,'train':train,'valid':valid,'test':test,'seed':1234,'runtime':{'torch':torch.__version__,'numpy':np.__version__,'device':device},'deviations':['modern runtime','single explicit NumPy sampler stream replaces eight worker streams','dense degree features','chunked head gradient accumulation'],'trace':trace}
 if best_state:summary['selected']=json.loads((output/'selected.json').read_text())
 (output/'result.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
