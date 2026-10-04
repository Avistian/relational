"""B20 course experiment: exposure order is the intervention within each pair."""
import hashlib
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# %% Learner mechanism 1: preserve the multiset, change only its ordering.
def exposure_order(levels, mode, seed, repeats=2):
    levels=np.asarray(levels)
    if levels.ndim!=1 or not len(levels) or mode not in ('staged','shuffled') or type(repeats) is not int or repeats<1:
        raise ValueError('Nonempty levels, staged/shuffled and positive integer repeats required')
    if not np.isfinite(levels).all():raise ValueError('Finite difficulty levels required')
    rng=np.random.default_rng(seed)
    if mode=='staged':
        return np.concatenate([rng.permutation(np.tile(np.flatnonzero(levels==level),repeats)) for level in np.unique(levels)])
    return rng.permutation(np.tile(np.arange(len(levels)),repeats))

# %% Learner mechanism 2: certify a pair before interpreting its difference.
def paired_contract(left, right):
    required=('pool','init','updates','optimizer','exposures','selection','evaluation','batch_size')
    for key in required:
        if key not in left or key not in right or left[key]!=right[key]:raise ValueError('Unmatched '+key)
    return True

# %% Learner mechanism 3: distinguish raw AUROC from excess above chance.
def auc_retention(score, baseline):
    if not np.isfinite([score,baseline]).all() or not 0<=score<=1 or not .5<baseline<=1:
        raise ValueError('Score in [0,1]; baseline strictly above chance')
    return dict(raw=score/baseline,above_chance=(score-.5)/(baseline-.5))

# %% Data: simple course priors, deliberately not the historical PluRel corpus.
def task_pool(seed, family, count=96):
    if family not in ('single','relational') or count%3:raise ValueError('Family and balanced three-stage count required')
    rng=np.random.default_rng(seed);xs=[];ys=[];cells=[]
    widths=np.repeat([2,4,6],count//3)
    for width in widths:
        parent=rng.normal(size=(32,3))
        if family=='single':
            x=rng.normal(size=(32,6));generated=32*int(width)
        else:
            # PKs 0..31; 96 child FKs. Aggregate only feature values, never targets.
            fk=rng.integers(0,32,96);values=parent[fk,0]+rng.normal(size=96)
            counts=np.bincount(fk,minlength=32);sums=np.bincount(fk,weights=values,minlength=32)
            means=sums/np.maximum(counts,1);maxima=np.zeros(32)
            for row in range(32):
                if counts[row]:maxima[row]=values[fk==row].max()
            x=np.column_stack([parent,counts,means,maxima]);generated=32*4+96*2 # count physical PK/FK too
        x[:,width:]=0
        weights=rng.normal(size=6);weights[width:]=0
        score=x@weights/np.sqrt(width)+rng.normal(scale=.15,size=32)
        # Threshold from support only. Query targets cannot determine the task boundary.
        y=(score>np.median(score[:16])).astype(np.int64)
        xs.append(x.astype(np.float32));ys.append(y);cells.append(generated)
    return dict(x=np.stack(xs),y=np.stack(ys),levels=widths,cells=np.asarray(cells))

def array_hash(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a);h.update(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes())
    return h.hexdigest()

# %% Model: source-shaped feature attention then support-only row attention.
def normalize_support(x, support):
    z=x.unsqueeze(-1);train=z[:,:support];valid=~torch.isnan(train)
    count=valid.sum(1,keepdim=True).clamp(min=1)
    mean=torch.where(valid,train,0.).sum(1,keepdim=True)/count
    var=torch.where(valid,train-mean,0.).square().sum(1,keepdim=True)/count
    return ((torch.where(torch.isnan(z),mean,z)-mean)/torch.sqrt(var+1e-20)).clamp(-100,100)

class FeatureEncoder(nn.Module):
    def __init__(self,width):
        super().__init__();self.linear_layer=nn.Linear(1,width)
    def forward(self,x,support):return self.linear_layer(normalize_support(x,support))

class TargetEncoder(nn.Module):
    def __init__(self,width):
        super().__init__();self.linear_layer=nn.Linear(1,width)
    def forward(self,y,rows):
        padding=y.float().mean(1,keepdim=True).repeat(1,rows-y.shape[1],1)
        return self.linear_layer(torch.cat([y,padding],1).unsqueeze(-1))

class BiAttention(nn.Module):
    def __init__(self,width,heads,hidden):
        super().__init__()
        self.self_attention_between_features=nn.MultiheadAttention(width,heads,batch_first=True)
        self.self_attention_between_datapoints=nn.MultiheadAttention(width,heads,batch_first=True)
        self.linear1=nn.Linear(width,hidden);self.linear2=nn.Linear(hidden,width)
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width);self.norm3=nn.LayerNorm(width)
    def forward(self,x,support):
        b,n,f,d=x.shape
        z=x.reshape(b*n,f,d)
        z=self.norm1((z+self.self_attention_between_features(z,z,z)[0]).reshape(b,n,f,d))
        z=z.transpose(1,2).reshape(b*f,n,d);keys=z[:,:support]
        left=self.self_attention_between_datapoints(keys,keys,keys)[0]
        right=self.self_attention_between_datapoints(z[:,support:],keys,keys)[0]
        z=self.norm2((z+torch.cat([left,right],1)).reshape(b,f,n,d).transpose(1,2))
        return self.norm3(z+self.linear2(F.gelu(self.linear1(z))))

class Decoder(nn.Module):
    def __init__(self,width,hidden):
        super().__init__();self.linear1=nn.Linear(width,hidden);self.linear2=nn.Linear(hidden,2)
    def forward(self,x):return self.linear2(F.gelu(self.linear1(x)))

class RDBPFN(nn.Module):
    """Compact course setting, not the paper's reported width/layers/checkpoint."""
    def __init__(self,width=16,heads=2,hidden=32,layers=2):
        super().__init__();self.feature_encoder=FeatureEncoder(width);self.target_encoder=TargetEncoder(width)
        self.transformer_blocks=nn.ModuleList([BiAttention(width,heads,hidden) for _ in range(layers)])
        self.decoder=Decoder(width,hidden)
    def forward(self,src,support):
        x,y=src
        if y.shape[1]!=support:raise ValueError('Pass support labels only')
        if y.ndim==2:y=y.unsqueeze(-1)
        z=torch.cat([self.feature_encoder(x,support),self.target_encoder(y,x.shape[1])],2)
        for block in self.transformer_blocks:z=block(z,support)
        return self.decoder(z[:,support:,-1])

# %% The same optimizer lives through every stage; fixed final checkpoint.
def fit_course(pool, mode, seed):
    torch.set_num_threads(1);torch.manual_seed(seed)
    model=RDBPFN();initial=array_hash(*[v.detach().numpy() for v in model.state_dict().values()])
    order=exposure_order(pool['levels'],mode,100+seed,2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    x=torch.from_numpy(pool['x']);y=torch.from_numpy(pool['y']);losses=[]
    for index in order:
        optimizer.zero_grad();logits=model((x[index:index+1],y[index:index+1,:16]),16)
        loss=F.cross_entropy(logits.reshape(-1,2),y[index,16:]);loss.backward();optimizer.step()
        losses.append(float(loss.detach()))
    return model,dict(init=initial,order=order.tolist(),losses=losses,optimizer_steps=int(next(iter(optimizer.state.values()))['step']))

def evaluate(model,pool):
    model.eval()
    with torch.no_grad():
        return model((torch.from_numpy(pool['x']),torch.from_numpy(pool['y'][:,:16])),16).softmax(-1)[...,1].numpy()
