"""Visible L086 operators and full PyG 1.2.0 citation-benchmark reconstruction.
Upstream MIT: pyg-team/pytorch_geometric d5aff37604c8e3247f5e807f2ba0ec6eeb4c661b.
Only unweighted undirected simple graphs are used for the GCN normalization.
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, HeteroData, Batch
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import remove_self_loops, add_self_loops


def normalized_edges(edge_index, num_nodes, dtype):
    """Replace self-loops, then compute symmetric endpoint-degree weights."""
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    src, dst = edge_index
    degree = torch.bincount(src, minlength=num_nodes).to(dtype)
    norm = degree[src].rsqrt() * degree[dst].rsqrt()
    return edge_index, norm


def weighted_messages(x_j, norm):
    """TODO 1: one scalar edge coefficient multiplies every message channel."""
    return norm[:, None] * x_j


class TraceGCN(MessagePassing):
    """Explicit transformation, source lifting, sum reduction and bias update."""
    def __init__(self, inputs, outputs, bias=False):
        super().__init__(aggr='add', flow='source_to_target')
        self.weight = nn.Parameter(torch.empty(inputs, outputs))
        self.bias = nn.Parameter(torch.zeros(outputs)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edge_index):
        edges, norm = normalized_edges(edge_index, x.size(0), x.dtype)
        return self.propagate(edges, x=x @ self.weight, norm=norm, size=(len(x),len(x)))

    def message(self, x_j, norm):
        return weighted_messages(x_j, norm)

    def update(self, aggr_out):
        return aggr_out if self.bias is None else aggr_out + self.bias


def seed_loss(logits, batch):
    """TODO 2: NeighborLoader places supervised seeds before context nodes."""
    return F.cross_entropy(logits[:batch.batch_size], batch.y[:batch.batch_size])


def global_edges(batch):
    """TODO 3: both edge rows contain local IDs; recover original graph IDs."""
    return batch.n_id[batch.edge_index]


def toy_data():
    return Data(x=torch.tensor([[2.],[4.],[8.],[10.]]),
                edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]]),
                y=torch.tensor([0,1,0,1]),num_nodes=4)


def loader_check():
    """Execute real sampling, including seed ordering, mapping and label isolation."""
    data=toy_data()
    batch=next(iter(NeighborLoader(data,input_nodes=torch.tensor([1]),
        num_neighbors=[-1,-1],batch_size=1,shuffle=False)))
    assert batch.batch_size==1 and batch.n_id[0].item()==1
    assert set(batch.n_id.tolist())=={0,1,2}
    torch.testing.assert_close(batch.x,data.x[batch.n_id])
    original=set(map(tuple,data.edge_index.t().tolist()))
    assert all(tuple(e) in original for e in global_edges(batch).t().tolist())
    logits=torch.randn(batch.num_nodes,2,requires_grad=True)
    before=seed_loss(logits,batch);before.backward()
    assert logits.grad[1:].abs().sum()==0 and logits.grad[0].abs().sum()>0
    altered=batch.clone();altered.y[1:]=1-altered.y[1:]
    torch.testing.assert_close(before,seed_loss(logits,altered))
    return {'n_id':batch.n_id.tolist(),'local_edges':batch.edge_index.tolist(),
            'global_edges':global_edges(batch).tolist(),'seed_only_gradient':'PASS'}


class CitationGCN(nn.Module):
    """Historical PyG benchmark: biases, hidden dropout only, two 16-wide layers."""
    def __init__(self, features, classes):
        super().__init__()
        self.conv1=TraceGCN(features,16,bias=True)
        self.conv2=TraceGCN(16,classes,bias=True)
    def forward(self,data):
        h=F.relu(self.conv1(data.x,data.edge_index))
        h=F.dropout(h,p=.5,training=self.training)
        return self.conv2(h,data.edge_index)


def as_data(loaded):
    """Use the same hash-verified Cora data and row normalization as L082."""
    x,s,y,train,valid,test=loaded
    row,col=s.indices();keep=row!=col
    data=Data(x=x.to_dense(),edge_index=torch.stack([col[keep],row[keep]]),y=y,num_nodes=x.size(0))
    for key,idx in [('train_mask',train),('val_mask',valid),('test_mask',test)]:
        mask=torch.zeros(len(y),dtype=torch.bool);mask[idx]=True;data[key]=mask
    return data


def train_citation(data,seed):
    """Full 200-epoch cap, stop after epoch 100, select smallest validation CE.
    Unlike old code, test labels are read only once after checkpoint selection.
    """
    torch.manual_seed(seed)
    model=CitationGCN(data.num_features,int(data.y.max())+1)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.0005)
    history=[];best=float('inf');chosen=None;best_epoch=None
    for epoch in range(1,201):
        model.train();optimizer.zero_grad()
        F.cross_entropy(model(data)[data.train_mask],data.y[data.train_mask]).backward()
        optimizer.step();model.eval()
        with torch.no_grad():vl=float(F.cross_entropy(model(data)[data.val_mask],data.y[data.val_mask]))
        history.append(vl)
        if vl<best:
            best=vl;chosen=copy.deepcopy(model.state_dict());best_epoch=epoch
        if epoch>100 and vl>torch.tensor(history[-11:-1]).mean().item():break
    model.load_state_dict(chosen);model.eval()
    with torch.no_grad():
        pred=model(data)[data.test_mask].argmax(1)
        accuracy=float((pred==data.y[data.test_mask]).float().mean())
    return {'seed':seed,'epochs':epoch,'selected_epoch':best_epoch,'validation_loss':history,'test_accuracy':accuracy}
