"""L090 inductive checkpoint extension; NOT a GraphSAGE paper-table reproduction.
The GCN lane uses the separately inlined L082 release-protocol implementation.
"""
import copy
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F


def eligible_neighbors(a, eligible):
    """Keep edges only when BOTH endpoints belong to the available context."""
    allowed=np.zeros(a.shape[0],dtype=bool);allowed[eligible]=True
    a=sp.csr_matrix(a)
    return [a.indices[a.indptr[i]:a.indptr[i+1]][allowed[a.indices[a.indptr[i]:a.indptr[i+1]]]]
            if allowed[i] else np.array([],dtype=int) for i in range(a.shape[0])]


def draw_neighbors(roots, neighbors, fanout, rng):
    """Uniform draws with replacement; empty rows use a masked root placeholder."""
    ids=np.repeat(np.asarray(roots)[:,None],fanout,axis=1)
    mask=np.zeros(ids.shape,dtype=bool)
    for row,node in enumerate(roots):
        if len(neighbors[node]):
            ids[row]=rng.choice(neighbors[node],size=fanout,replace=True);mask[row]=True
    return ids,mask


def masked_mean(values,mask):
    """values [B,K,F], validity [B,K] -> [B,F]; an empty neighborhood gives zero."""
    weights=mask.to(values.dtype).unsqueeze(-1)
    return (values*weights).sum(1)/weights.sum(1).clamp_min(1)


def verdict(mean,n,protocol_aligned,target=.815,tolerance=.01):
    """Course tolerance is meaningful only after declared protocol and coverage pass."""
    if not protocol_aligned:return 'INCOMPARABLE'
    if n!=100:return 'INCOMPLETE'
    return 'CLOSE' if abs(mean-target)<=tolerance else 'FAIL'


class SampledSAGE(nn.Module):
    """Two shared root/neighbor transforms; hidden ReLU and L2 normalization.
    Width32, biased output logits, no dropout: a declared educational variant.
    """
    def __init__(self,features,classes):
        super().__init__();self.first=nn.Linear(2*features,32);self.second=nn.Linear(64,classes)
    def hidden(self,own,neighbor_mean):
        return F.normalize(F.relu(self.first(torch.cat([own,neighbor_mean],dim=-1))),p=2,dim=-1)
    def sampled(self,x,roots,neighbors,rng,fanout=10):
        near,m1=draw_neighbors(roots,neighbors,fanout,rng)
        far,m2=draw_neighbors(near.reshape(-1),neighbors,fanout,rng)
        own_hidden=self.hidden(x[roots],masked_mean(x[near],torch.tensor(m1)))
        near_hidden=self.hidden(x[near.reshape(-1)],masked_mean(x[far],torch.tensor(m2)))
        mean_hidden=masked_mean(near_hidden.reshape(len(roots),fanout,32),torch.tensor(m1))
        return self.second(torch.cat([own_hidden,mean_hidden],dim=-1))
    def full(self,x,neighbors):
        # Sparse row mean, no self-loop: self is the separate concatenation branch.
        rows=[];cols=[];values=[]
        for i,ns in enumerate(neighbors):
            if not len(ns):continue
            rows.extend([i]*len(ns));cols.extend(ns);values.extend([1/len(ns)]*len(ns))
        s=torch.sparse_coo_tensor(np.array([rows,cols],dtype=np.int64),torch.tensor(values,dtype=x.dtype),(len(x),len(x))).coalesce()
        h=self.hidden(x,torch.sparse.mm(s,x))
        return self.second(torch.cat([h,torch.sparse.mm(s,h)],dim=-1))


def inductive_contexts(data):
    x,s,y,train,valid,test=data
    indices=s.indices().numpy();keep=indices[0]!=indices[1]
    a=sp.coo_matrix((np.ones(keep.sum()),indices[:,keep]),shape=s.shape).tocsr()
    base=np.setdiff1d(np.arange(len(y)),np.r_[valid.numpy(),test.numpy()])
    return (x.to_dense(),y,train,valid,test,
            eligible_neighbors(a,base),eligible_neighbors(a,np.r_[base,valid.numpy()]),
            eligible_neighbors(a,np.r_[base,test.numpy()]))


def train_inductive(data,seed,epochs=100):
    """Select by validation only; reveal test nodes after restoring selected weights."""
    torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    x,y,train,valid,test,nt,nv,ne=inductive_contexts(data)
    model=SampledSAGE(x.shape[1],int(y.max())+1)
    opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.0005)
    best=float('inf');history=[];state=None;chosen=0
    for epoch in range(epochs):
        model.train();order=rng.permutation(train.numpy())
        for start in range(0,len(order),64):
            roots=order[start:start+64];opt.zero_grad()
            F.cross_entropy(model.sampled(x,roots,nt,rng),y[roots]).backward();opt.step()
        model.eval()
        with torch.no_grad():vl=float(F.cross_entropy(model.full(x,nv)[valid],y[valid]))
        history.append(vl)
        if vl<best:best=vl;chosen=epoch+1;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    with torch.no_grad():
        predictions=model.full(x,ne)[test].argmax(1)
        accuracy=float((predictions==y[test]).float().mean())
    return {'seed':seed,'epochs':epochs,'selected_epoch':chosen,'validation_loss':history,
            'test_accuracy':accuracy,'test_predictions':predictions.tolist(),
            'train_context_nodes':len(y)-len(valid)-len(test),'labeled_train_nodes':len(train),
            'batch_size':64,'fanouts':[10,10],'protocol':'inductive Cora extension; not paper comparable'}
