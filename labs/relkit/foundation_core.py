"""Visible mechanisms for L061–L069; explicitly reduced teaching architectures.

RowPFN mirrors label-conditioned row attention. AxialPFN mirrors alternating
feature/context attention. Neither is an official TabPFN checkpoint architecture.
Every algorithm is small enough to inspect and is inlined into the relevant lab.
"""
import math
import numpy as np
import torch
from torch import nn


def posterior_predictive(labels, alpha=1., beta=1.):
    """Beta–Bernoulli posterior predictive probability of the next success."""
    y = np.asarray(labels)
    if alpha <= 0 or beta <= 0 or not np.isin(y, [0, 1]).all():
        raise ValueError('Positive prior counts and binary labels required')
    return float((alpha + y.sum()) / (alpha + beta + len(y)))


class CountPFN(nn.Module):
    """PFN objective with an exact sufficient-statistic encoder, not TabPFN.

    Input [successes, failures]/32; output next-label logit. This deliberately
    removes representation learning so L061 can test posterior approximation.
    """
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2,32),nn.GELU(),nn.Linear(32,32),nn.GELU(),nn.Linear(32,1))

    def forward(self,counts):
        return self.net(counts/32.).squeeze(-1)


def sample_coin_tasks(rng,batch=256,max_context=32):
    """Context counts and query label share one latent probability per task."""
    theta=rng.beta(1.,1.,batch)
    n=rng.integers(1,max_context+1,batch)
    success=rng.binomial(n,theta)
    query=rng.binomial(1,theta)
    return np.column_stack([success,n-success]).astype('float32'),query.astype('float32')


def context_mask(n_context, n_query):
    """Boolean allow-mask: every row reads context; no row reads a query key."""
    if n_context < 1 or n_query < 1:
        raise ValueError('A nonempty context and query are required')
    mask = torch.zeros(n_context + n_query, n_context + n_query, dtype=torch.bool)
    mask[:, :n_context] = True
    return mask


def attention(q, k, v, allowed=None):
    """Scaled dot-product attention; True means allowed (PyTorch SDPA convention)."""
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if allowed is not None:
        if not allowed.any(dim=-1).all():
            raise ValueError('Each query needs an allowed key')
        scores = scores.masked_fill(~allowed, float('-inf'))
    return scores.softmax(-1) @ v


class AttentionBlock(nn.Module):
    """Pre-normalized multihead attention plus residual feed-forward block."""
    def __init__(self, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError('Width must be divisible by heads')
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.out = nn.Linear(width, 3*width), nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))

    def forward(self, x, allowed=None):
        b, n, d = x.shape
        q, k, v = [a.reshape(b,n,self.heads,d//self.heads).transpose(1,2)
                   for a in self.qkv(self.norm1(x)).chunk(3,-1)]
        h = attention(q,k,v,allowed).transpose(1,2).reshape(b,n,d)
        x = x + self.out(h)
        return x + self.ff(self.norm2(x))


class RowPFN(nn.Module):
    """Reduced row-token PFN. Labels are embedded only for context rows.

    Shapes: context [B,C,F], query [B,Q,F], labels [B,C], logits [B,Q,2].
    No positions: simultaneous context row/label permutation leaves predictions fixed.
    """
    def __init__(self, features, width=32, heads=4, layers=2):
        super().__init__()
        self.x_encoder = nn.Linear(features,width)
        self.y_encoder = nn.Embedding(3,width)  # 0,1 plus an explicit unknown symbol
        self.blocks = nn.ModuleList([AttentionBlock(width,heads) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2))

    def embeddings(self, context, labels, query):
        c, q = context.shape[1], query.shape[1]
        h = self.x_encoder(torch.cat([context,query],1))
        unknown = torch.full((len(context),q),2,dtype=torch.long,device=context.device)
        h = h + self.y_encoder(torch.cat([labels.long(),unknown],1))
        allow = context_mask(c,q).to(context.device)
        for block in self.blocks:
            h = block(h,allow)
        return h[:,c:]

    def forward(self, context, labels, query):
        return self.head(self.embeddings(context,labels,query))


class AxialPFN(nn.Module):
    """Key-part v2 mirror: feature attention then context-only sample attention.

    One scalar feature per token, separate target token, two classes. Omits feature
    grouping/IDs, distribution encoders, repeated ensembling and official prior.
    """
    def __init__(self, features, width=32, heads=4, layers=2):
        super().__init__()
        self.features=features
        self.x_encoder=nn.Linear(1,width)
        self.y_encoder=nn.Embedding(3,width)
        self.feature_blocks=nn.ModuleList([AttentionBlock(width,heads) for _ in range(layers)])
        self.sample_blocks=nn.ModuleList([AttentionBlock(width,heads) for _ in range(layers)])
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2))

    def embeddings(self,context,labels,query):
        b,c,f=context.shape;q=query.shape[1];n=c+q
        x=self.x_encoder(torch.cat([context,query],1).unsqueeze(-1))
        unknown=torch.full((b,q),2,dtype=torch.long,device=context.device)
        y=self.y_encoder(torch.cat([labels.long(),unknown],1)).unsqueeze(2)
        h=torch.cat([x,y],2);d=h.shape[-1];g=f+1
        allow=context_mask(c,q).to(context.device)
        for feature,sample in zip(self.feature_blocks,self.sample_blocks):
            h=feature(h.reshape(b*n,g,d)).reshape(b,n,g,d)
            h=axial_sample_attention(sample,h,c)
        return h[:,c:,-1]

    def forward(self,context,labels,query):
        return self.head(self.embeddings(context,labels,query))


def axial_sample_attention(block,h,n_context):
    """Keep features separate, attend over rows, then restore [B,N,F,D]."""
    b,n,f,d=h.shape
    allow=context_mask(n_context,n-n_context).to(h.device)
    rows=h.permute(0,2,1,3).reshape(b*f,n,d)
    out=block(rows,allow)
    return out.reshape(b,f,n,d).permute(0,2,1,3)


def induced_column(tokens, inducing, n_context):
    """TabICL Eq.4/5 attention skeleton (no learned projections/residual/FFN).

    Training keys compress to m inducing vectors; all rows read the m summaries.
    Query values must not affect another query or any context embedding.
    """
    if not 0 < n_context <= len(tokens):
        raise ValueError('Invalid context length')
    memory=attention(inducing,tokens[:n_context],tokens[:n_context])
    return attention(tokens,memory,memory)


def distribution_embedding(values, tokens, inducing, n_context, weight_head, bias_head):
    """TabICL Eq.1/2: each cell gets a distribution-conditioned affine map."""
    h=induced_column(tokens,inducing,n_context)
    return weight_head(h)*values[...,None]+bias_head(h)


def sample_scm(noise, weights, activation=np.tanh):
    """Topological evaluation. weights[parent,child]; strictly upper triangular DAG."""
    noise,weights=np.asarray(noise,float),np.asarray(weights,float)
    d=noise.shape[1]
    if weights.shape!=(d,d) or not np.allclose(np.tril(weights),0):
        raise ValueError('Weights must encode a strictly upper-triangular DAG')
    values=np.empty_like(noise)
    for child in range(d):
        signal=values[:,:child] @ weights[:child,child]
        values[:,child]=activation(signal)+noise[:,child]
    return values


def observe_scm(values,feature_nodes,target_node,threshold):
    """Declare observed variables and a fixed threshold; reject target-as-feature."""
    if target_node in feature_nodes or len(set(feature_nodes))!=len(feature_nodes):
        raise ValueError('Distinct feature nodes must exclude the target')
    return values[:,feature_nodes],(values[:,target_node]>threshold).astype('int64')


def nearest_context(train, queries, k, exclude_ids=None):
    """Stable exact squared-Euclidean neighbors after train-only preprocessing."""
    train,queries=np.asarray(train),np.asarray(queries)
    if k<1 or k>len(train)-(exclude_ids is not None):
        raise ValueError('Context size exceeds eligible memory')
    distance=((queries[:,None,:]-train[None,:,:])**2).sum(-1)
    if exclude_ids is not None:
        ids=np.asarray(exclude_ids)
        if ids.shape!=(len(queries),) or (ids<0).any() or (ids>=len(train)).any():
            raise ValueError('One valid excluded training-row ID per query required')
        distance[np.arange(len(queries)),ids]=np.inf
    return np.argsort(distance,axis=1,kind='stable')[:,:k]


def local_episode(x,y,anchor,context_size,query_size,rng):
    """LoCalPFN §2.4 neighborhood approximation: disjoint context/query near an anchor."""
    neighbors=nearest_context(x,x[[anchor]],context_size+query_size,np.array([anchor]))[0]
    shuffled=rng.permutation(neighbors)
    c,q=shuffled[:context_size],shuffled[context_size:]
    return x[c],y[c],x[q],y[q],c,q


def temporal_eligible(event_time,label_time,cutoff):
    """Both clocks must be observed by cutoff; equality is allowed by this contract."""
    return (np.asarray(event_time)<=cutoff)&(np.asarray(label_time)<=cutoff)


def crossfit_embeddings(x,y,fold_ids,embed):
    """Write each row's query-role representation using other folds as labeled context."""
    folds=np.asarray(fold_ids)
    if len(np.unique(folds))<2 or len(folds)!=len(x):
        raise ValueError('At least two aligned folds required')
    result=None
    for fold in np.unique(folds):
        query=folds==fold;context=~query
        h=np.asarray(embed(x[context],y[context],x[query]))
        if result is None:result=np.empty((len(x),h.shape[1]),dtype=h.dtype)
        result[query]=h
    return result


def open_class_loss(labels, probabilities, classes, epsilon=1e-12):
    """Score every row, including unseen labels assigned zero supported probability.

    epsilon gives a finite diagnostic penalty; disclose it instead of dropping rows.
    """
    probabilities=np.asarray(probabilities,float)
    if probabilities.shape!=(len(labels),len(classes)) or not np.allclose(probabilities.sum(1),1):
        raise ValueError('Aligned normalized probability matrix required')
    lookup={v:i for i,v in enumerate(classes)}
    assigned=[probabilities[i,lookup[y]] if y in lookup else 0. for i,y in enumerate(labels)]
    return float(-np.log(np.clip(assigned,epsilon,1)).mean())


def corrupt_column(train,test,column,mode):
    """Frozen test-only intervention; choose center from training and never mutate inputs."""
    result=np.asarray(test).copy();center=np.asarray(train)[:,column].mean()
    if mode=='missing':result[:,column]=center
    elif mode=='scale':result[:,column]=center+3*(result[:,column]-center)
    else:raise ValueError('Choose missing or scale')
    return result
