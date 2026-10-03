"""Visible Griffin operators and checkpoint-compatible release model."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def cell_attention(q,k,v,blocked=None,dropout=0.,training=False):
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError('Q/K width and K/V cell count must align')
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if blocked is not None:
        if blocked.dtype != torch.bool:
            raise ValueError('blocked must be boolean')
        blocked = torch.broadcast_to(blocked, scores.shape)
        if blocked.all(-1).any():
            raise ValueError('Each query needs at least one visible cell')
        scores = scores.masked_fill(blocked, -torch.inf)
    weights = scores.softmax(-1)
    return F.dropout(weights, p=dropout, training=training) @ v

def relation_pool(x,edge_index,relation,relation_vectors):
    if edge_index is None or edge_index.numel() == 0:
        return x * 0
    receiver, sender = edge_index
    pairs, inverse = torch.unique(torch.stack((receiver, relation)), dim=1, return_inverse=True)
    total = torch.zeros((pairs.shape[1], x.shape[1]), dtype=x.dtype, device=x.device)
    total.index_add_(0, inverse, x[sender])
    count = torch.bincount(inverse, minlength=pairs.shape[1]).to(x.dtype)
    messages = (total / count[:, None]) * relation_vectors[pairs[1]]
    out = torch.full_like(x, -torch.inf)
    out.scatter_reduce_(0, pairs[0, :, None].expand_as(messages), messages, reduce='amax', include_self=True)
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))

def eligible_edges(times,owner_cutoffs):
    if times.ndim != 2 or owner_cutoffs.shape != (times.shape[0],):
        raise ValueError('One cutoff per owner; no implicit batch-wide broadcast')
    return (times != -1) & (times < owner_cutoffs[:, None])

class CellAttention(nn.Module):
    """Checkpoint names match release; kernel math is visible and learner-editable."""
    def __init__(self,hiddim,first=False):
        super().__init__();self.first=first
        self.linq=nn.Linear(hiddim,hiddim,bias=False)
        self.crossattention=nn.MultiheadAttention(hiddim,8,dropout=.1,bias=False,batch_first=True)

    def forward(self,metadata,values,task=None,blocked=None):
        b,c,d=values.shape;heads=8;width=d//heads
        keys=metadata.unsqueeze(0).expand(b,-1,-1)
        queries=keys if self.first else task.unsqueeze(1)
        wq,wk,wv=self.crossattention.in_proj_weight.chunk(3,dim=0)
        def split(x,w):return F.linear(x,w).reshape(b,-1,heads,width).transpose(1,2)
        mask=None if blocked is None else blocked[:,None,None,:]
        out=cell_attention(split(queries,wq),split(keys,wk),split(values,wv),mask,.1,self.training)
        out=out.transpose(1,2).reshape(b,-1,d)
        out=F.linear(out,self.crossattention.out_proj.weight)
        if self.first:return self.linq(out).mean(1)
        return (out*self.linq(queries)).squeeze(1)

class RelationMessage(nn.Module):
    def __init__(self,hiddim):
        super().__init__();self.rellin=nn.Sequential(nn.Linear(hiddim,hiddim))

    def forward(self,x,edge_index,relation,metadata):
        if edge_index is None or edge_index.numel()==0:
            return 0*self.rellin(x[0])  # exact release broadcast path
        return relation_pool(x,edge_index,relation,self.rellin(metadata))

class GriffinMod(nn.Module):
    """Released four-layer family, including optional gates and reverse messages.

    Frozen text/float encoders and root sampling live outside this module. Inputs
    are lists of (column metadata[C,D], values[N,C,D]), masks and task vectors.
    The classification decoder is explicit in classification_logits below.
    """
    def __init__(self,hiddim=512,num_mp=4,use_rev=True,use_gate=False):
        super().__init__();self.num_mp=num_mp;self.use_rev=use_rev;self.use_gate=use_gate
        self.nodefeataggr=nn.ModuleList([CellAttention(hiddim,first=i==0) for i in range(num_mp)])
        self.mpnn=nn.ModuleList([RelationMessage(hiddim) for _ in range(num_mp)])
        self.lintask=nn.ModuleList([nn.Linear(hiddim,hiddim,bias=False) for _ in range(num_mp-1)])
        self.mlp=nn.ModuleList([nn.Sequential(nn.Linear(hiddim,hiddim,bias=False),nn.SiLU(inplace=True)) for _ in range(num_mp)])
        self.mlp2=nn.ModuleList([nn.Sequential(nn.Linear(hiddim,hiddim,bias=False),nn.SiLU(inplace=True),nn.Linear(hiddim,hiddim,bias=False)) for _ in range(num_mp)])
        self.ln=nn.LayerNorm(hiddim,elementwise_affine=False)
        def gates():
            layers=nn.ModuleList([nn.Sequential(nn.Linear(hiddim,int(hiddim**.5)),nn.SiLU(inplace=True),nn.Linear(int(hiddim**.5),1,bias=False)) for _ in range(num_mp)])
            with torch.no_grad():
                for layer in layers:layer[2].weight.zero_()
            return layers
        self.gatelin=gates() if use_gate else None
        self.revmpnn=nn.ModuleList([RelationMessage(hiddim) for _ in range(num_mp)]) if use_rev else None
        self.revgatelin=gates() if use_rev and use_gate else None

    def forward(self,node,mask,taskfeat,edge_index,edge_attr_type,edge_attr):
        tasks=[]
        for task,(metadata,values) in zip(taskfeat,node):
            t=metadata.mean(0) if task is None else task
            tasks.append(t.expand(values.shape[0],-1) if t.ndim==1 else t)
        counts=[v.shape[0] for _,v in node]
        for layer in range(self.num_mp):
            cells=torch.cat([self.nodefeataggr[layer](m,v,self.ln(t),blocked) for (m,v),t,blocked in zip(node,tasks,mask)],dim=0)
            x=cells if layer==0 else x+cells
            normal=self.ln(x);messages=self.mlp[layer](normal)
            forward=self.mpnn[layer](messages,edge_index,edge_attr_type,edge_attr)
            if self.use_gate:forward=forward*self.gatelin[layer](normal)
            reverse=0
            if self.use_rev:
                reverse=self.revmpnn[layer](messages,None if edge_index is None else edge_index[[1,0]],edge_attr_type,edge_attr)
                if self.use_gate:reverse=reverse*self.revgatelin[layer](normal)
            x=x+self.mlp2[layer](normal)+forward+reverse
            if layer<self.num_mp-1:
                updates=self.lintask[layer](self.ln(x)).split(counts)
                tasks=[old+new for old,new in zip(tasks,updates)]
        return self.ln(x)

def classification_logits(model,batch):
    node,mask,tasks,edges,relations,edge_vectors,labels,label_vectors,root_mapping=batch
    if label_vectors is None:raise ValueError('Classification requires candidate label embeddings')
    return model(node,mask,tasks,edges,relations,edge_vectors)[root_mapping] @ label_vectors.T
