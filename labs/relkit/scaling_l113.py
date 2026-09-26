"""Visible OGB products ClusterGCN replay; MIT-derived architecture, see sources/l113.
Partitioning is provided METIS infrastructure. Aggregation, model, loss,
selection, training, and exact layerwise inference are visible here.
"""
# %% Imports and frozen experiment
import copy
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
TARGETS = {'valid':92.12, 'test':78.97}
TOLERANCE_PP = .5
PAPER = {'layers':3,'hidden':256,'dropout':.5,'lr':.001,'epochs':50,
         'partitions':15000,'clusters_per_batch':32,'seeds':list(range(10)),
         'evaluation_epochs':list(range(20,51,5))}

# %% Task 1: incoming neighbor mean

def mean_adjacency(edge_index, num_nodes):
    """Input edges [source, receiver]; output CSR [receiver, source]. No loops added."""
    src,dst=edge_index
    coo=torch.sparse_coo_tensor(torch.stack([dst,src]),torch.ones(src.numel(),device=src.device),(num_nodes,num_nodes)).coalesce()
    degree=torch.zeros(num_nodes,device=src.device).scatter_add_(0,coo.indices()[0],coo.values())
    values=coo.values()/degree[coo.indices()[0]].clamp_min(1)
    return torch.sparse_coo_tensor(coo.indices(),values,coo.shape).coalesce().to_sparse_csr()

# %% Task 2: induced subgraph and local identities

def induced_edges(edge_index, node_ids, num_nodes):
    """Retain exactly edges with both ends in the batch, then relabel locally."""
    mapping=torch.full((num_nodes,),-1,dtype=torch.long,device=edge_index.device)
    mapping[node_ids]=torch.arange(len(node_ids),device=edge_index.device)
    local=mapping[edge_index]
    return local[:,(local>=0).all(dim=0)]

# %% GraphSAGE: mean neighbors plus separate root transform
class MeanSAGE(nn.Module):
    def __init__(self,inputs,outputs):
        super().__init__()
        self.lin_l=nn.Linear(inputs,outputs,bias=True)
        self.lin_r=nn.Linear(inputs,outputs,bias=False)
    def forward(self,x,adj,root=None):
        if root is None:root=x
        return self.lin_l(torch.sparse.mm(adj,x))+self.lin_r(root)

class SAGE(nn.Module):
    def __init__(self,inputs=100,hidden=256,outputs=47,layers=3,dropout=.5):
        super().__init__()
        dims=[inputs]+[hidden]*(layers-1)+[outputs]
        self.convs=nn.ModuleList([MeanSAGE(a,b) for a,b in zip(dims[:-1],dims[1:])])
        self.dropout=dropout
    def forward(self,x,adj):
        for i,conv in enumerate(self.convs):
            x=conv(x,adj)
            if i<len(self.convs)-1:x=F.dropout(F.relu(x),self.dropout,self.training)
        return F.log_softmax(x,dim=-1)

def masked_loss(log_prob,y,mask):
    if not bool(mask.any()):raise ValueError('Batch has no training labels; skip optimizer step')
    return F.nll_loss(log_prob[mask],y[mask])

# %% Task 3: select only by validation, preserving the first tie

def selected_epoch(history):
    if not history:raise ValueError('No evaluated epochs')
    return max(range(len(history)),key=lambda i:history[i]['valid'])

# %% Exact inference in receiver chunks, one full layer at a time
@torch.no_grad()
def layerwise_inference(model,x,adj,batch_size=65536,device='cpu'):
    """All neighbors, no sampling; adjacency stays on CPU. Each layer completes first.
    Only the previous full feature matrix is resident on device (2.34 GiB at width256).
    Global output row order is preserved, including isolated nodes.
    """
    model.eval();rowptr=adj.crow_indices();cols=adj.col_indices();values=adj.values()
    for i,conv in enumerate(model.convs):
        previous=x.to(device);out=torch.empty((len(x),conv.lin_l.out_features),dtype=x.dtype)
        for start in range(0,len(x),batch_size):
            end=min(len(x),start+batch_size);lo=int(rowptr[start]);hi=int(rowptr[end])
            block=torch.sparse_csr_tensor((rowptr[start:end+1]-lo).to(device),cols[lo:hi].to(device),values[lo:hi].to(device),size=(end-start,len(x)))
            z=conv(previous,block,previous[start:end])
            if i<len(model.convs)-1:z=F.relu(z)
            out[start:end]=z.cpu()
        x=out;del previous
    return x

# %% Metrics, fingerprints and recorded runtime

def scores_from_predictions(pred,y,split):
    return {k:float((pred[idx]==y[idx]).float().mean()) for k,idx in split.items()}

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def tensor_hash(x):
    return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()

# %% Complete selected schedule, with a hard runtime boundary

def train_run(data,cluster_data,adj,split,output,seed=0,epochs=50,device='cpu',max_seconds=2400,workers=0):
    """No automatic resume. Save selected inference checkpoint plus all evaluated scores.
    A cutoff yields INCOMPLETE; it is never counted as a complete seed.
    """
    from torch_geometric.loader import ClusterLoader
    output=Path(output)
    if output.exists():raise FileExistsError('Use a fresh output directory: '+str(output))
    output.mkdir(parents=True)
    torch.manual_seed(seed);np.random.seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats()
    model=SAGE(data.x.shape[1],256,int(data.y.max())+1,3,.5).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    loader=ClusterLoader(cluster_data,batch_size=32,shuffle=True,num_workers=workers)
    history=[];epoch_times=[];best=-math.inf;start=time.perf_counter();status='COMPLETE'
    y=data.y.view(-1)
    for epoch in range(1,epochs+1):
        model.train();tick=time.perf_counter();loss_sum=0.;examples=0
        for batch in loader:
            if time.perf_counter()-start>max_seconds:status='INCOMPLETE';break
            mask=batch.train_mask.to(device)
            if not bool(mask.any()):continue
            a=mean_adjacency(batch.edge_index,batch.num_nodes).to(device)
            xb=batch.x.to(device);yb=batch.y.view(-1).to(device)
            optimizer.zero_grad();logp=model(xb,a);loss=masked_loss(logp,yb,mask)
            loss.backward();optimizer.step()
            n=int(mask.sum());loss_sum+=float(loss.detach())*n;examples+=n
        if device!='cpu':torch.cuda.synchronize()
        epoch_times.append({'epoch':epoch,'seconds':time.perf_counter()-tick,'loss':loss_sum/max(examples,1),'training_examples':examples,'finished':status=='COMPLETE'})
        print(json.dumps(epoch_times[-1]),flush=True)
        if status!='COMPLETE':break
        # Short pilot gets one exact evaluation; published schedule remains 20,25,...,50.
        evaluate=(epoch>=20 and epoch%5==0) or (epochs<20 and epoch==epochs)
        if evaluate:
            tick=time.perf_counter();logits=layerwise_inference(model,data.x,adj,65536,device)
            pred=logits.argmax(-1);metrics=scores_from_predictions(pred,y,split)
            row={'epoch':epoch,**metrics,'inference_seconds':time.perf_counter()-tick};history.append(row)
            if metrics['valid']>best:
                best=metrics['valid'];torch.save(model.state_dict(),output/'checkpoint.pt')
                np.savez_compressed(output/'predictions.npz',pred=pred.numpy().astype(np.int16))
            print(json.dumps(row),flush=True)
        (output/'progress.json').write_text(json.dumps({'epochs':epoch_times,'history':history},indent=2))
    selected=history[selected_epoch(history)] if history else None
    result={'status':status,'seed':seed,'requested_epochs':epochs,'epochs':epoch_times,'history':history,'selected':selected,
            'seconds':time.perf_counter()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device!='cpu' else 0,
            'scope':'PUBLISHED_SCHEDULE' if epochs==50 else 'PILOT_ONLY','trainable_parameters':sum(p.numel() for p in model.parameters())}
    (output/'result.json').write_text(json.dumps(result,indent=2))
    return result
