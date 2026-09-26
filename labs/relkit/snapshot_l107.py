"""Readable GCN-GRU and EvolveGCN release port (IBM, Apache-2.0, 3f4996a).
The release O cell is a matrix GRU, NOT the LSTM described in paper section 3.5.
"""
# %% Imports
import math
import numpy as np
import torch
from torch import nn

# %% TODO 1: normalize a completed snapshot

def normalized_adjacency(pairs, weights, n):
    """Add self loops, coalesce repeated edges, use row-degree normalization."""
    pairs=np.asarray(pairs,dtype=np.int64).reshape(-1,2)
    idx=torch.as_tensor(np.concatenate([pairs,np.stack([np.arange(n)]*2,axis=1)]).T)
    val=torch.cat([torch.as_tensor(weights,dtype=torch.float32),torch.ones(n)])
    a=torch.sparse_coo_tensor(idx,val,(n,n)).coalesce()
    degree=torch.zeros(n).scatter_add_(0,a.indices()[0],a.values())
    i,j=a.indices();v=a.values()/torch.sqrt(degree[i]*degree[j])
    return torch.sparse_coo_tensor(a.indices(),v,(n,n)).coalesce()

# %% TODO 2: evolve a matrix state

def matrix_update(previous, summary, gates):
    """Paper section 3.4: gate=1 selects the candidate (opposite PyTorch GRU z)."""
    update,reset,candidate=gates
    z=update(summary,previous);r=reset(summary,previous)
    proposed=candidate(summary,r*previous)
    return (1-z)*previous+z*proposed

# %% Released objective

def weighted_cross_entropy(logits, labels, weights):
    """IBM Cross_Entropy.py: divide by number of examples, not sum of weights."""
    return (weights[labels]*(-logits.gather(1,labels[:,None]).flatten()+torch.logsumexp(logits,dim=1))).mean()

# %% Matrix gates and learned top-k summary

class MatrixGate(nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__();self.activation=activation
        self.W=nn.Parameter(torch.empty(rows,rows));nn.init.uniform_(self.W,-1/math.sqrt(rows),1/math.sqrt(rows))
        self.U=nn.Parameter(torch.empty(rows,rows));nn.init.uniform_(self.U,-1/math.sqrt(rows),1/math.sqrt(rows))
        self.bias=nn.Parameter(torch.zeros(rows,cols))
    def forward(self,x,h):return self.activation(self.W@x+self.U@h+self.bias)

class TopSummary(nn.Module):
    def __init__(self,rows,k):
        super().__init__();self.k=k;self.scorer=nn.Parameter(torch.empty(rows,1))
        nn.init.uniform_(self.scorer,-1/math.sqrt(rows),1/math.sqrt(rows))
    def forward(self,x,mask):
        scores=x@self.scorer/self.scorer.norm()+mask
        values,indices=scores.flatten().topk(self.k)
        indices=indices[values>-float('inf')]
        if not len(indices):raise ValueError('Release TopK is undefined with no active nodes')
        if len(indices)<self.k:indices=torch.cat([indices,indices[-1].repeat(self.k-len(indices))])
        return (x[indices]*torch.tanh(scores[indices])).T

class EvolvingLayer(nn.Module):
    def __init__(self,rows,cols,variant):
        super().__init__();self.variant=variant
        self.gates=nn.ModuleList([MatrixGate(rows,cols,torch.sigmoid),MatrixGate(rows,cols,torch.sigmoid),MatrixGate(rows,cols,torch.tanh)])
        # Retain the O release's unused scorer so parameter initialization consumes the same RNG.
        self.summary=TopSummary(rows,cols)
        self.initial=nn.Parameter(torch.empty(rows,cols));nn.init.uniform_(self.initial,-1/math.sqrt(cols),1/math.sqrt(cols))
    def forward(self,adj,features,masks):
        w=self.initial;outputs=[]
        for a,x,mask in zip(adj,features,masks):
            q=self.summary(x,mask) if self.variant=='H' else w
            w=matrix_update(w,q,self.gates)
            # Source RReLU stays in training mode during validation too.
            outputs.append(nn.functional.rrelu(a@(x@w),lower=1/8,upper=1/3,training=True))
        return outputs

class EvolveGCN(nn.Module):
    def __init__(self,features,hidden,variant):
        super().__init__();self.layers=nn.ModuleList([EvolvingLayer(features,hidden,variant),EvolvingLayer(hidden,hidden,variant)])
    def forward(self,adj,features,masks):
        for layer in self.layers:features=layer(adj,features,masks)
        return features[-1]

class PairClassifier(nn.Module):
    def __init__(self,dim,hidden):
        super().__init__();self.mlp=nn.Sequential(nn.Linear(2*dim,hidden),nn.ReLU(),nn.Linear(hidden,2))
    def forward(self,z,pairs):return self.mlp(torch.cat([z[pairs[:,0]],z[pairs[:,1]]],1))

# %% TODO 3: legal snapshot cutoff

def completed_bins(times,width,query):
    """[k*w,(k+1)*w) is visible only after its scheduled close. a=t assumed."""
    if width<=0:raise ValueError('width must be positive')
    bins=np.floor(np.asarray(times)/width).astype(np.int64)
    return bins[(bins+1)*width<=query]

# %% Course GCN followed by node-state GRU

class SnapshotGRU(nn.Module):
    def __init__(self,input_dim,hidden=32):
        super().__init__();self.input_dim=input_dim;self.hidden=hidden
        self.gcn1=nn.Linear(input_dim,hidden,bias=False);self.gcn2=nn.Linear(hidden,hidden,bias=False)
        self.gru=nn.GRUCell(hidden,hidden);self.decoder=PairClassifier(hidden,hidden)
    def step(self,a,x,state):
        z=torch.relu(a@self.gcn1(x));z=torch.relu(a@self.gcn2(z))
        return self.gru(z,state)
    def score(self,state,pairs):return self.decoder(state,pairs)
