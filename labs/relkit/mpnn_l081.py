"""L081: generic message routing and a reconstruction of the released sparse GG-NN.

Equations follow brain-research/mpnn commit 4a1f0ddea3cd7de5eebc96e509da2161624aaacd.
This is a PyTorch reconstruction, not a TensorFlow checkpoint/runtime replay.
"""
import torch
from torch import nn
from torch.nn import functional as F


def aggregate(messages, destination, n_nodes, reduction='sum'):
    """Reduce [E,D] edge messages to [N,D]; isolated nodes receive zero."""
    if reduction not in ('sum','mean'):
        raise ValueError('Use sum or mean')
    out=messages.new_zeros(n_nodes,messages.shape[1])
    out.index_add_(0,destination,messages)
    if reduction=='mean':
        degree=torch.bincount(destination,minlength=n_nodes).clamp_min(1)
        out=out/degree[:,None]
    return out


def mean_step(h, edge_index):
    """Synchronous L078 update: half self + half incoming-neighbor mean."""
    src,dst=edge_index
    messages=h[src]
    incoming=aggregate(messages,dst,len(h),'mean')
    return .5*h+.5*incoming


def graph_sum(h, batch):
    """Invariant graph readout; batch[i] is the graph ID of node i."""
    return aggregate(h,batch,int(batch.max())+1,'sum')


class GenericMPNN(nn.Module):
    """Inject M(h_destination,h_source,e), U(h,m), and R(h,batch)."""
    def __init__(self,message,update,readout,reduction='sum',steps=1):
        super().__init__();self.message=message;self.update=update
        self.readout=readout;self.reduction=reduction;self.steps=steps

    def forward(self,h,edge_index,edge_attr,batch):
        src,dst=edge_index
        for _ in range(self.steps):
            m=self.message(h[dst],h[src],edge_attr)
            h=self.update(h,aggregate(m,dst,len(h),self.reduction))
        return h,self.readout(h,batch)


class SourceGRU(nn.Module):
    """Released bias-free gates; reset acts BEFORE recurrent multiplication.

    torch.nn.GRUCell uses a different candidate equation and is not substituted.
    """
    def __init__(self,width):
        super().__init__()
        for name,shape in [('w_z',(2*width,width)),('u_z',(width,width)),
                           ('w_r',(2*width,width)),('u_r',(width,width)),
                           ('w',(2*width,width)),('u',(width,width))]:
            p=nn.Parameter(torch.empty(shape));nn.init.xavier_uniform_(p)
            setattr(self,name,p)

    def forward(self,h,m):
        z=torch.sigmoid(m@self.w_z+h@self.u_z)
        r=torch.sigmoid(m@self.w_r+h@self.u_r)
        candidate=torch.tanh(m@self.w+(r*h)@self.u)
        return (1-z)*h+z*candidate


class SparseGGNN(nn.Module):
    """Four bond types, two message banks, tied recurrence, gated graph readout.

    x: [N,13] Table-1-style atom features. Edges: both orientations of each bond.
    No distance, virtual edge, self edge, or explicit hydrogen nodes in this lane.
    """
    def __init__(self,input_dim=13,width=50,steps=6,readout_width=200):
        super().__init__()
        if width<input_dim:raise ValueError('Hidden width must fit zero-padded atom features')
        self.width=width;self.steps=steps
        self.bond_in=nn.Parameter(torch.empty(4,width,width))
        self.bond_out=nn.Parameter(torch.empty(4,width,width))
        for bank in (self.bond_in,self.bond_out):
            for matrix in bank:nn.init.xavier_uniform_(matrix)
        self.update=SourceGRU(width)
        self.gate=nn.Sequential(nn.Linear(width+input_dim,readout_width),nn.ReLU(),nn.Linear(readout_width,1))
        self.value=nn.Sequential(nn.Linear(width+input_dim,readout_width),nn.ReLU(),nn.Linear(readout_width,1))
        for layer in [*self.gate,*self.value]:
            if isinstance(layer,nn.Linear):nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)

    def message(self,h,edge_index,edge_type):
        src,dst=edge_index
        # Matrices act on column states, as in the released dense operator.
        incoming=torch.bmm(self.bond_in[edge_type],h[src,None].transpose(1,2)).squeeze(-1)
        outgoing=torch.bmm(self.bond_out[edge_type],h[src,None].transpose(1,2)).squeeze(-1)
        return torch.cat([aggregate(incoming,dst,len(h)),aggregate(outgoing,dst,len(h))],1)

    def forward(self,x,edge_index,edge_type,batch):
        h=F.pad(x,(0,self.width-x.shape[1]))
        for _ in range(self.steps):h=self.update(h,self.message(h,edge_index,edge_type))
        combined=torch.cat([h,x],1)
        return graph_sum(torch.sigmoid(self.gate(combined))*self.value(combined),batch).flatten()
