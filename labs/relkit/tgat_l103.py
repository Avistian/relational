"""Readable TGAT release-protocol reconstruction for lesson 103.
Reference: Xu et al., ICLR 2020; release 9293d10d1943c4bd4a186337cf38ba98e4c8bb99.
This is an independently structured, focused implementation of the attention variant.
"""
# %% Imports and frozen identities
import copy
import hashlib
import json
import math
import random
import time
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score
SOURCE_COMMIT='9293d10d1943c4bd4a186337cf38ba98e4c8bb99'
DATA_URL='https://snap.stanford.edu/jodie/wikipedia.csv'
DATA_SHA256='a6b73e09c0d1e5b9db11e7e7aa416f2e87838a745273e0446a79952cf4cfae09'
PAPER_AP={'all':95.34,'new':93.99}
CLOSE_TOLERANCE_PP=0.5

# %% Task 1: functional time encoding, release parameterization

def encode_time(elapsed, frequency, phase):
    """Map elapsed [B,N] and trainable [D] frequencies/phases to [B,N,D]."""
    return torch.cos(elapsed.unsqueeze(-1)*frequency+phase)

# %% Task 2: scaled masked attention

def attention_weights(query, keys, padding):
    """[H*B,1,D] x [H*B,N,D] -> [H*B,1,N]; True means padding.
    Preserve the release's finite -1e10 mask, including uniform all-padding rows.
    """
    scores=torch.bmm(query,keys.transpose(1,2))/math.sqrt(query.shape[-1])
    return torch.softmax(scores.masked_fill(padding,-1e10),dim=-1)

# %% Task 3: strict-past temporal boundary for the corrected comparison

def strict_prefix(sorted_times, cutoff):
    """Number of events strictly before cutoff; exclude every equal timestamp."""
    return int(np.searchsorted(sorted_times,cutoff,side='left'))

# %% Temporal neighborhoods: declared release behavior or corrected boundary

class NeighborFinder:
    def __init__(self, events, n_nodes, release=True, uniform=True):
        rows=[[] for _ in range(n_nodes)]
        for u,v,t,e in zip(events['u'],events['v'],events['t'],events['e']):
            rows[int(u)].append((int(v),int(e),float(t)))
            rows[int(v)].append((int(u),int(e),float(t)))
        # Release sorts by edge ID. For this stream, IDs are chronological.
        self.rows=[np.asarray(sorted(r,key=lambda z:z[1] if release else z[2]),dtype=np.float64).reshape(-1,3) for r in rows]
        self.release,self.uniform=release,uniform
    def find_before(self,node,cutoff):
        a=self.rows[int(node)];ts=a[:,2]
        if not self.release:
            end=strict_prefix(ts,cutoff)
        elif not len(a):
            end=0
        else:
            # Exact released binary-search/slice convention, including its omitted item.
            left,right=0,len(a)-1
            while left+1<right:
                middle=(left+right)//2
                if ts[middle]<cutoff:left=middle
                else:right=middle
            end=right if ts[right]<cutoff else left
        return a[:end]
    def get_temporal_neighbor(self,nodes,cutoffs,num_neighbors=20):
        ids=np.zeros((len(nodes),num_neighbors),dtype=np.int32)
        edges=np.zeros_like(ids);times=np.zeros(ids.shape,dtype=np.float32)
        for i,(node,cutoff) in enumerate(zip(nodes,cutoffs)):
            history=self.find_before(node,cutoff)
            if not len(history):continue
            if self.uniform:
                chosen=history[np.random.randint(0,len(history),num_neighbors)]
                ids[i],edges[i],times[i]=chosen[:,0],chosen[:,1],chosen[:,2]
                order=times[i].argsort();ids[i],edges[i],times[i]=ids[i,order],edges[i,order],times[i,order]
            else:
                chosen=history[:num_neighbors] if self.release else history[-num_neighbors:]
                ids[i,-len(chosen):],edges[i,-len(chosen):],times[i,-len(chosen):]=chosen[:,0],chosen[:,1],chosen[:,2]
        return ids,edges,times

# %% Trainable encoder and feed-forward merge

class TimeEncoder(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.basis_freq=nn.Parameter(torch.tensor(1/10**np.linspace(0,9,dim),dtype=torch.float32))
        self.phase=nn.Parameter(torch.zeros(dim))
    def forward(self,elapsed):return encode_time(elapsed,self.basis_freq,self.phase)

class Merge(nn.Module):
    def __init__(self,left,right,hidden,out):
        super().__init__();self.fc1=nn.Linear(left+right,hidden);self.fc2=nn.Linear(hidden,out)
        nn.init.xavier_normal_(self.fc1.weight);nn.init.xavier_normal_(self.fc2.weight)
    def forward(self,left,right):return self.fc2(torch.relu(self.fc1(torch.cat([left,right],dim=-1))))

# %% Multi-head attention: projection, aggregation, residual and normalization

class MultiHead(nn.Module):
    def __init__(self,width,heads,dropout):
        super().__init__();assert width%heads==0
        self.heads,self.head_dim=heads,width//heads
        self.w_qs=nn.Linear(width,width,bias=False);self.w_ks=nn.Linear(width,width,bias=False);self.w_vs=nn.Linear(width,width,bias=False)
        for layer in [self.w_qs,self.w_ks,self.w_vs]:nn.init.normal_(layer.weight,std=np.sqrt(2/(width+self.head_dim)))
        self.layer_norm=nn.LayerNorm(width);self.fc=nn.Linear(width,width);nn.init.xavier_normal_(self.fc.weight)
        self.dropout=nn.Dropout(dropout)
    def forward(self,q,k,mask):
        batch=q.shape[0]
        def split(x,projection):
            return projection(x).view(batch,-1,self.heads,self.head_dim).permute(2,0,1,3).contiguous().view(self.heads*batch,-1,self.head_dim)
        query,keys,values=split(q,self.w_qs),split(k,self.w_ks),split(k,self.w_vs)
        weights=self.dropout(attention_weights(query,keys,mask.repeat(self.heads,1,1)))
        mixed=torch.bmm(weights,values).view(self.heads,batch,1,self.head_dim).permute(1,2,0,3).contiguous().view(batch,1,-1)
        return self.layer_norm(q+self.dropout(self.fc(mixed)))

class TemporalAttention(nn.Module):
    def __init__(self,dim,heads,dropout):
        super().__init__();self.merger=Merge(3*dim,dim,dim,dim);self.multi_head_target=MultiHead(3*dim,heads,dropout)
    def forward(self,source,zero_time,neighbors,elapsed,edge,padding):
        q=torch.cat([source[:,None,:],torch.zeros_like(source[:,None,:]),zero_time],dim=-1)
        kv=torch.cat([neighbors,edge,elapsed],dim=-1)
        context=self.multi_head_target(q,kv,padding[:,None,:]).squeeze(1)
        return self.merger(context,source)

# %% Complete recursive TGAT encoder and shared positive/negative decoder

class TGAT(nn.Module):
    def __init__(self,finder,node_features,edge_features,layers=2,heads=2,dropout=.1):
        super().__init__();self.num_layers=layers;self.ngh_finder=finder
        # Preserve release registration and initialization order for parity checks.
        self.n_feat_th=nn.Parameter(torch.as_tensor(node_features,dtype=torch.float32))
        self.e_feat_th=nn.Parameter(torch.as_tensor(edge_features,dtype=torch.float32))
        self.edge_raw_embed=nn.Embedding.from_pretrained(self.e_feat_th,padding_idx=0,freeze=True)
        self.node_raw_embed=nn.Embedding.from_pretrained(self.n_feat_th,padding_idx=0,freeze=True)
        d=node_features.shape[1];assert edge_features.shape[1]==d
        self.merge_layer=Merge(d,d,d,d) # unused released module, retained for RNG/state identity
        self.attn_model_list=nn.ModuleList([TemporalAttention(d,heads,dropout) for _ in range(layers)])
        self.time_encoder=TimeEncoder(d);self.affinity_score=Merge(d,d,d,1)
    def tem_conv(self,nodes,cutoffs,layer,num_neighbors=20):
        device=self.n_feat_th.device
        raw=self.node_raw_embed(torch.as_tensor(nodes,dtype=torch.long,device=device))
        if layer==0:return raw
        source=self.tem_conv(nodes,cutoffs,layer-1,num_neighbors)
        nbr,eid,when=self.ngh_finder.get_temporal_neighbor(nodes,cutoffs,num_neighbors)
        child=self.tem_conv(nbr.ravel(),when.ravel(),layer-1,num_neighbors).reshape(len(nodes),num_neighbors,-1)
        elapsed=torch.as_tensor(cutoffs[:,None]-when,dtype=torch.float32,device=device)
        time_features=self.time_encoder(elapsed)
        zero_time=self.time_encoder(torch.zeros((len(nodes),1),device=device))
        edge=self.edge_raw_embed(torch.as_tensor(eid,dtype=torch.long,device=device))
        padding=torch.as_tensor(nbr==0,device=device)
        return self.attn_model_list[layer-1](source,zero_time,child,time_features,edge,padding)
    def contrast(self,u,v,negative,t,num_neighbors=20):
        src=self.tem_conv(u,t,self.num_layers,num_neighbors)
        dst=self.tem_conv(v,t,self.num_layers,num_neighbors)
        neg=self.tem_conv(negative,t,self.num_layers,num_neighbors)
        return self.affinity_score(src,dst).flatten().sigmoid(),self.affinity_score(src,neg).flatten().sigmoid()

# %% Hash-checked raw Wikipedia preprocessing and frozen temporal split
def load_wikipedia(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory/'wikipedia.csv'
    if not path.exists():
        urllib.request.urlretrieve(DATA_URL, path)
    sha = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    assert sha == DATA_SHA256, f'Dataset checksum mismatch: {sha}'
    cache = directory/'processed.npz'
    if cache.exists():
        z = np.load(cache)
        # The raw identity alone cannot authenticate an arbitrary processed cache.
        if str(z['raw_sha']) != sha:
            raise ValueError('Processed cache belongs to other raw bytes')
        u,v,t,x = (z[k] for k in ['u','v','t','x'])
    else:
        # Header has four columns, rows additionally have 172 edge features.
        a = np.loadtxt(path, delimiter=',', skiprows=1)
        u = a[:, 0].astype(np.int64)+1
        v = a[:, 1].astype(np.int64)+int(u.max())+1
        t,x = a[:, 2], a[:, 4:].astype(np.float32)
        np.savez(cache, u=u,v=v,t=t,x=x,raw_sha=sha)
    assert len(u)==157474 and x.shape==(157474,172)
    assert np.all(np.diff(t)>=0) and len(set(u)|set(v))==9227
    edges = np.vstack([np.zeros((1,172),np.float32),x])
    nodes = np.zeros((int(max(u.max(),v.max()))+1,172),np.float32)
    full = {'u':u,'v':v,'t':t,'e':np.arange(1,len(u)+1)}
    val_time,test_time = np.quantile(t,[.70,.85])
    candidates = set(u[t>val_time]) | set(v[t>val_time])
    # Python 3.9 random.sample(set, k) internally converted to tuple. Preserve that order.
    held = set(random.Random(2020).sample(tuple(candidates),int(.1*(len(nodes)-1))))
    train_mask = (t<=val_time)&~np.isin(u,list(held))&~np.isin(v,list(held))
    seen = set(u[train_mask])|set(v[train_mask]); unseen=(set(u)|set(v))-seen
    novel = np.isin(u,list(unseen))|np.isin(v,list(unseen))
    masks={'train':train_mask,'val':(t>val_time)&(t<=test_time),'test':t>test_time}
    masks['new_val']=masks['val']&novel; masks['new_test']=masks['test']&novel
    data={name:{k:a[mask] for k,a in full.items()} for name,mask in masks.items()}
    data['full']=full
    audit={'raw_sha256':sha,'cutoffs':[float(val_time),float(test_time)],
           'counts':{k:len(d['u']) for k,d in data.items()},'held_nodes':sorted(int(i) for i in held),
           'split_sha256':{k:hashlib.sha256(d['e'].tobytes()).hexdigest() for k,d in data.items()}}
    return nodes,edges,data,audit

# %% Release negative sampling, batch boundaries and metrics

class NegativeSampler:
    def __init__(self,events):self.u=np.unique(events['u']);self.v=np.unique(events['v'])
    def sample(self,n):
        np.random.randint(0,len(self.u),n) # release consumes an unused source draw
        return self.v[np.random.randint(0,len(self.v),n)]

def event_batches(events,size,release=True):
    n=len(events['u'])
    for start in range(0,n,size):
        end=min(n-1 if release else n,start+size)
        if end<=start:continue # avoid empty-batch crash; original Wikipedia has no empty tail
        yield {k:x[start:end] for k,x in events.items()}

@torch.no_grad()
def evaluate(model,events,sampler,neighbors=20,batch_size=30,release=True):
    model.eval();rows=[];saved=[]
    for b in event_batches(events,batch_size,release):
        neg=sampler.sample(len(b['u']));p,n=model.contrast(b['u'],b['v'],neg,b['t'],neighbors)
        p,n=p.cpu().numpy(),n.cpu().numpy();scores=np.r_[p,n];truth=np.r_[np.ones(len(p)),np.zeros(len(n))]
        rows.append([average_precision_score(truth,scores),roc_auc_score(truth,scores),np.mean((scores>.5)==truth)])
        saved.append({'e':b['e'],'negative':neg,'p':p,'n':n,'batch':np.full(len(p),len(rows)-1)})
    return dict(zip(['ap','auc','accuracy'],np.mean(rows,axis=0).tolist())),saved

def compact_weights(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k not in ['n_feat_th','e_feat_th','edge_raw_embed.weight','node_raw_embed.weight']}

# %% Complete released-experiment training and checkpoint selection

def run_training(nodes,edges,data,seed=0,epochs=50,output=None,device='cpu',neighbors=20,layers=2,release=True):
    """Complete modern release replay. Historical exact seed identity is unavailable."""
    torch.manual_seed(seed);np.random.seed(seed)
    train=data['train'];train_finder=NeighborFinder(train,len(nodes),release);full_finder=NeighborFinder(data['full'],len(nodes),release)
    model=TGAT(train_finder,nodes,edges,layers=layers).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.0001)
    train_sampler=NegativeSampler(train);val_sampler=NegativeSampler(data['full'])
    idx=np.arange(len(train['u']));np.random.shuffle(idx) # retained even though release never uses idx
    output=Path(output or f'l103-paper/seed-{seed}');output.mkdir(parents=True,exist_ok=True)
    trace=[];best=None;best_epoch=0;rounds=0;stopped=False
    for epoch in range(epochs):
        begin=time.perf_counter();np.random.shuffle(idx);model.ngh_finder=train_finder;model.train();losses=[]
        for b in event_batches(train,200,release):
            optimizer.zero_grad();p,n=model.contrast(b['u'],b['v'],train_sampler.sample(len(b['u'])),b['t'],neighbors)
            loss=nn.functional.binary_cross_entropy(p,torch.ones_like(p))+nn.functional.binary_cross_entropy(n,torch.zeros_like(n))
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        model.ngh_finder=full_finder
        val,_=evaluate(model,data['val'],val_sampler,neighbors,release=release)
        new_val,_=evaluate(model,data['new_val'],val_sampler,neighbors,release=release)
        if best is None:best=val['ap']
        elif (val['ap']-best)/abs(best)>.001:
            best=val['ap'];rounds=0;best_epoch=epoch+1 if release else epoch
        else:rounds+=1
        row={'epoch':epoch,'loss':float(np.mean(losses)),'val_ap':val['ap'],'new_val_ap':new_val['ap'],'seconds':time.perf_counter()-begin}
        trace.append(row);print(json.dumps({'seed':seed,**row}),flush=True)
        (output/'progress.json').write_text(json.dumps(trace,indent=2))
        if rounds>=3:
            model.load_state_dict(torch.load(output/f'epoch-{best_epoch}.pt',map_location=device,weights_only=True),strict=False)
            stopped=True;break
        torch.save(compact_weights(model),output/f'epoch-{epoch}.pt')
    selected=best_epoch if stopped else len(trace)-1
    # Save the RNG state immediately before evaluation for an exact source replay.
    np_state=np.random.get_state();torch_state=torch.get_rng_state()
    torch.save({'weights':compact_weights(model),'numpy_state':np_state,'torch_state':torch_state},output/'selected.pt')
    test,pred=evaluate(model,data['test'],NegativeSampler(data['full']),neighbors,release=release)
    new_test,new_pred=evaluate(model,data['new_test'],NegativeSampler(data['new_test']),neighbors,release=release)
    np.savez_compressed(output/'predictions.npz',**{f'{lane}_{k}':np.concatenate([x[k] for x in records]) for lane,records in [('all',pred),('new',new_pred)] for k in records[0]})
    result={'seed':seed,'epochs_completed':len(trace),'selected_epoch':selected,'early_stopped':stopped,'test':test,'new_test':new_test,'trace':trace,'release_quirks':release,'neighbors':neighbors,'layers':layers,'train_events':len(train['u'])-(1 if release else 0),'test_events':sum(len(x['e']) for x in pred),'new_test_events':sum(len(x['e']) for x in new_pred)}
    (output/'result.json').write_text(json.dumps(result,indent=2));return result
