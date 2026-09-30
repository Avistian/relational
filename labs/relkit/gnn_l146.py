"""Course GNN: typed row/time/type encoding, relation means and root readout.

This explicit control is not the published RDL HeteroGraphSAGE configuration.
Each destination type has one self transform; each directed FK relation has
one neighbor transform. Different relation means are summed, then normalized.
"""
import torch
from torch import nn
from relkit.relgt_course_l146 import NeighborTfsEncoder,NeighborTimeEncoder

def relation_mean(x, edges, n):
    out=x.new_zeros((n,x.shape[-1]));count=x.new_zeros((n,1))
    out.index_add_(0,edges[1],x[edges[0]])
    count.index_add_(0,edges[1],x.new_ones((edges.shape[1],1)))
    return out/count.clamp_min(1)

class TypedMeanLayer(nn.Module):
    def __init__(self,width,types,relations,dropout):
        super().__init__()
        self.self_linears=nn.ModuleList([nn.Linear(width,width) for _ in range(types)])
        self.neighbor_linears=nn.ModuleList([nn.Linear(width,width,bias=False) for _ in range(relations)])
        self.norm=nn.LayerNorm(width);self.dropout=nn.Dropout(dropout)
    def forward(self,x,node_types,edges,relation):
        out=torch.zeros_like(x)
        for t,linear in enumerate(self.self_linears):
            mask=node_types==t;out[mask]=linear(x[mask])
        for r,linear in enumerate(self.neighbor_linears):
            e=edges[:,relation==r]
            if e.numel():out=out+linear(relation_mean(x,e,len(x)))
        return self.dropout(torch.relu(self.norm(out)))

class CourseGNN(nn.Module):
    def __init__(self,col_names_dict,stats,type_map,relations,width=64,layers=2,dropout=.1):
        super().__init__()
        self.row=NeighborTfsEncoder(width,type_map,col_names_dict,stats)
        self.time=NeighborTimeEncoder(width);self.type_embedding=nn.Embedding(len(type_map),width)
        self.norm=nn.LayerNorm(width)
        self.layers=nn.ModuleList([TypedMeanLayer(width,len(type_map),relations,dropout) for _ in range(layers)])
        self.head=nn.Sequential(nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))
    def forward(self,b):
        t=b['neighbor_types'];bs,k=t.shape
        x=self.norm(self.row(b,t)+self.type_embedding(t)+self.time(b['neighbor_times']))
        x=x.reshape(bs*k,-1)
        for layer in self.layers:x=layer(x,t.reshape(-1),b['edge_index'],b['relation'])
        return self.head(x.reshape(bs,k,-1)[:,0]).view(-1)
