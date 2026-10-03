"""Exact selected L074 CARTE definitions; source parity audited separately."""
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import PowerTransformer,StandardScaler

def make_graph(row, vectors):
    """A source-oriented star: index[0] receives, index[1] supplies each message."""
    leaves,edges=[],[]
    for column,value in row.items():
        if pd.isna(value):continue
        edge=np.asarray(vectors[column],dtype=np.float32)
        node=np.asarray(vectors[str(value).lower()],dtype=np.float32) if isinstance(value,str) else float(value)*edge
        leaves.append(node);edges.append(edge)
    if not leaves:raise ValueError('All-missing row: no observed leaf; choose an explicit fallback')
    leaves=torch.tensor(np.stack(leaves));edges=torch.tensor(np.stack(edges))
    center=(leaves*edges).mean(0,keepdim=True)
    x=torch.cat([center,leaves]);n=len(leaves);ids=torch.arange(1,n+1)
    index=torch.stack([torch.cat([torch.zeros(n,dtype=torch.long),ids]),torch.cat([ids,ids])])
    attrs=torch.cat([edges,torch.ones_like(edges)])
    return x,index,attrs

def grouped_attention(edge_index, query, key, value):
    """Normalize over incoming messages separately for each receiving node."""
    receiver=edge_index[0]
    logits=(query[receiver]*key).sum(-1)/math.sqrt(query.shape[-1])
    maxima=torch.full((len(query),),-torch.inf,dtype=query.dtype,device=query.device)
    maxima=maxima.scatter_reduce(0,receiver,logits,reduce='amax',include_self=True)
    weights=(logits-maxima[receiver]).exp()
    totals=torch.zeros_like(maxima).index_add(0,receiver,weights)
    weights=weights/totals[receiver]
    output=torch.zeros_like(query).index_add(0,receiver,weights[:,None]*value)
    return output,weights

class Attention(nn.Module):
    def __init__(self,d=300,heads=12):
        super().__init__();self.heads=heads
        self.lin_query=nn.Linear(d,d,bias=False)
        self.lin_key=nn.Linear(d,d,bias=False)
        self.lin_value=nn.Linear(d,d,bias=False)
    def forward(self,x,index,edge):
        z=edge*x[index[1]]
        q,k,v=self.lin_query(x),self.lin_key(z),self.lin_value(z)
        width=q.shape[1]//self.heads
        return torch.cat([grouped_attention(index,q[:,i:i+width],k[:,i:i+width],v[:,i:i+width])[0] for i in range(0,q.shape[1],width)],1)

class Readout(nn.Module):
    def __init__(self):
        super().__init__();self.g_attn=Attention()
        self.linear_net_x=nn.Sequential(nn.Linear(300,300),nn.Dropout(0),nn.GELU(),nn.Linear(300,300))
        self.norm1_x=nn.LayerNorm(300);self.norm2_x=nn.LayerNorm(300)
    def forward(self,x,index,edge):
        # The pinned code has no node residual addition in this block.
        x=self.norm1_x(self.g_attn(x,index,edge))
        return self.norm2_x(self.linear_net_x(x))

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_x=nn.Sequential(nn.Linear(300,300),nn.GELU(),nn.LayerNorm(300))
        self.initial_e=nn.Sequential(nn.Linear(300,300),nn.GELU(),nn.LayerNorm(300))
        self.read_out_block=Readout()
    def forward(self,x,index,edge):
        return self.read_out_block(self.initial_x(x),index,self.initial_e(edge))
    def rows(self,x,index,edge,heads):
        # Only center outputs are consumed. Leaf inputs still supply keys/values.
        receiver_map=torch.full((len(x),),-1,dtype=torch.long)
        receiver_map[heads]=torch.arange(len(heads))
        keep=receiver_map[index[0]]>=0
        row_ids=receiver_map[index[0,keep]]
        x=self.initial_x(x);edge=self.initial_e(edge[keep])
        attn=self.read_out_block.g_attn
        z=edge*x[index[1,keep]]
        q,k,v=attn.lin_query(x[heads]),attn.lin_key(z),attn.lin_value(z)
        row_index=torch.stack([row_ids,torch.zeros_like(row_ids)])
        width=q.shape[1]//attn.heads
        output=torch.cat([grouped_attention(row_index,q[:,i:i+width],k[:,i:i+width],v[:,i:i+width])[0] for i in range(0,q.shape[1],width)],1)
        output=self.read_out_block.norm1_x(output)
        return self.read_out_block.norm2_x(self.read_out_block.linear_net_x(output))

def load_encoder(checkpoint,pretrained,seed):
    torch.manual_seed(seed);model=Encoder()
    if pretrained:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        selected={k:state['ft_base.'+k] for k in model.state_dict()}
        model.load_state_dict(selected,strict=True)
    return model

def batch_graphs(graphs):
    xs,es,attrs,heads=[],[],[],[];offset=0
    for x,index,edge in graphs:
        heads.append(offset);xs.append(x);es.append(index+offset);attrs.append(edge);offset+=len(x)
    return torch.cat(xs),torch.cat(es,1),torch.cat(attrs),torch.tensor(heads)

def embed(model,graphs):
    model.eval();parts=[]
    with torch.no_grad():
        for i in range(0,len(graphs),64):
            x,index,edge,heads=batch_graphs(graphs[i:i+64]);parts.append(model.rows(x,index,edge,heads).numpy())
    return np.concatenate(parts)

def normalize_numeric(frame,train):
    """Fit per-column maps using train only; constant columns need no power fit."""
    out=frame.copy();policy={}
    for col in frame.select_dtypes(include='number'):
        observed=frame.iloc[train][col].dropna()
        if len(observed)==0:
            out[col]=np.nan;policy[col]='unobserved_train_omit';continue
        constant=observed.nunique()==1
        transformer=StandardScaler() if constant else PowerTransformer()
        transformer.fit(frame.iloc[train][[col]])
        out[col]=transformer.transform(frame[[col]]).flatten()
        policy[col]='standard_constant' if constant else 'power'
    return out,policy
