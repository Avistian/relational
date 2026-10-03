"""Visible L166 mechanisms. Course prior is deliberately small; model matches base release."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def relational_prior(seed, parents=8, children=24, strength=1.0):
    """Course two-table SCM; not the released LayerDAG/selective-SCM generator."""
    if any(type(v) is not int or v<1 for v in [parents,children]) or not np.isfinite(strength):
        raise ValueError('Positive integer sizes and finite strength required')
    rng=np.random.default_rng(seed)
    latent=rng.normal(size=parents)
    foreign_keys=rng.integers(parents,size=children)
    noise=rng.normal(scale=.2,size=children)
    return dict(parent_ids=np.arange(parents),latent=latent,child_parent=foreign_keys,
                noise=noise,values=strength*latent[foreign_keys]+noise)

def dfs_summary(parent_ids, child_parent, values):
    """One backward aggregation: COUNT and MEAN in the supplied parent order."""
    ids=list(parent_ids);fk=list(child_parent);v=np.asarray(values,dtype=float)
    if len(set(ids))!=len(ids) or len(fk)!=len(v) or v.ndim!=1 or not np.isfinite(v).all() or not set(fk)<=set(ids):
        raise ValueError('Unique parents, valid foreign keys and finite aligned values required')
    out=np.zeros((len(ids),2));lookup={key:i for i,key in enumerate(ids)}
    for key,value in zip(fk,v):
        out[lookup[key],0]+=1;out[lookup[key],1]+=value
    out[:,1]/=np.maximum(out[:,0],1)
    return out

def context_mask(rows, support):
    """True means a key/value is visible: all receivers read support only."""
    if type(rows) is not int or type(support) is not int or not 0<support<=rows:
        raise ValueError('Require 0 < support <= rows, both integers')
    return np.broadcast_to(np.arange(rows)<support,(rows,rows)).copy()

def normalize_support(x, support):
    """Released population-variance normalization before scalar-to-vector projection."""
    if x.ndim!=3 or not 0<support<=x.shape[1]:raise ValueError('B x N x F with nonempty support')
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
        # Query slots contain the support mean, never held-out query labels.
        padding=y.to(torch.float).mean(1,keepdim=True).repeat(1,rows-y.shape[1],1)
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
        z=z.transpose(1,2).reshape(b*f,n,d)
        keys=z[:,:support]
        left=self.self_attention_between_datapoints(keys,keys,keys)[0]
        right=self.self_attention_between_datapoints(z[:,support:],keys,keys)[0]
        z=self.norm2((z+torch.cat([left,right],1)).reshape(b,f,n,d).transpose(1,2))
        return self.norm3(z+self.linear2(F.gelu(self.linear1(z))))

class Decoder(nn.Module):
    def __init__(self,width,hidden):
        super().__init__();self.linear1=nn.Linear(width,hidden);self.linear2=nn.Linear(hidden,2)
    def forward(self,x):return self.linear2(F.gelu(self.linear1(x)))

class RDBPFN(nn.Module):
    """Checkpoint-compatible numeric base: six blocks, 96 channels, four heads."""
    def __init__(self,width=96,heads=4,hidden=192,layers=6):
        super().__init__();self.feature_encoder=FeatureEncoder(width);self.target_encoder=TargetEncoder(width)
        self.transformer_blocks=nn.ModuleList([BiAttention(width,heads,hidden) for _ in range(layers)])
        self.decoder=Decoder(width,hidden)
    def forward(self,src,support):
        x,y=src
        if y.ndim==2:y=y.unsqueeze(-1)
        z=torch.cat([self.feature_encoder(x,support),self.target_encoder(y,x.shape[1])],2)
        for block in self.transformer_blocks:z=block(z,support)
        return self.decoder(z[:,support:,-1])
