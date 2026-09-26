# %% Imports and frozen experiment
"""L112: full-batch OGB GCN, with visible graph algebra and selection.
Architecture follows OGB's MIT-licensed gnn.py; see labs/sources/l112/LICENSE.
"""
import copy
import hashlib
import json
import math
import random
import time
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from ogb.nodeproppred import Evaluator

DATA_URL='https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip'
DATA_SHA256='49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276'
TARGETS={'valid':73.00,'test':71.74}
TOLERANCE_PP=0.5

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

# %% Task 1: normalized message passing

def normalized_adjacency(edge_index, num_nodes):
    """Binary undirected graph + one self loop, then D^-1/2 A D^-1/2.

    Matrix rows are receivers and columns are senders. Deduplicate BEFORE
    measuring degrees: reciprocal citations must not get extra weight.
    """
    e=edge_index.cpu().long()
    loops=torch.arange(num_nodes).repeat(2,1)
    idx=torch.cat([e,e.flip(0),loops],dim=1)
    a=torch.sparse_coo_tensor(idx,torch.ones(idx.shape[1]),(num_nodes,num_nodes)).coalesce()
    row,col=a.indices()
    degree=torch.bincount(row,minlength=num_nodes).float()
    inv=degree.rsqrt()
    values=inv[row]*inv[col]
    return torch.sparse_coo_tensor(a.indices(),values,a.shape).coalesce().to_sparse_csr()

# %% Complete GCN architecture

class GraphConvolution(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(out_channels,in_channels))
        self.bias=nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    def forward(self,x,adj):
        return torch.sparse.mm(adj,F.linear(x,self.weight))+self.bias

class GCN(nn.Module):
    """128 -> 256 -> 256 -> 40; batch norm includes all graph nodes."""
    def __init__(self,in_channels=128,hidden_channels=256,out_channels=40,dropout=.5):
        super().__init__()
        self.convs=nn.ModuleList([GraphConvolution(in_channels,hidden_channels),GraphConvolution(hidden_channels,hidden_channels),GraphConvolution(hidden_channels,out_channels)])
        self.bns=nn.ModuleList([nn.BatchNorm1d(hidden_channels),nn.BatchNorm1d(hidden_channels)])
        self.dropout=dropout
    def reset_parameters(self):
        for conv in self.convs:conv.reset_parameters()
        for bn in self.bns:bn.reset_parameters()
    def forward(self,x,adj):
        for conv,bn in zip(self.convs[:-1],self.bns):
            x=F.dropout(F.relu(bn(conv(x,adj))),p=self.dropout,training=self.training)
        return self.convs[-1](x,adj).log_softmax(dim=-1)

# %% Task 2: train on authorized labels

def training_loss(log_probs, labels, train_idx):
    return F.nll_loss(log_probs[train_idx],labels[train_idx])

# %% Task 3: validation chooses the checkpoint

def selected_epoch(validation_scores):
    """Zero-based index, first maximizer, matching released torch.argmax."""
    if not len(validation_scores):raise ValueError('No evaluated epochs')
    return int(np.argmax(validation_scores))

# %% Raw release, official split files, data integrity

def load_arxiv(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);archive=root/'arxiv.zip'
    if not archive.exists():
        temp=archive.with_suffix('.partial');urllib.request.urlretrieve(DATA_URL,temp);temp.replace(archive)
    actual=sha256(archive)
    if actual!=DATA_SHA256:raise ValueError(f'Dataset hash mismatch: {actual}')
    with zipfile.ZipFile(archive) as z:
        def read(name,dtype):
            with z.open('arxiv/'+name) as f:return pd.read_csv(f,header=None,compression='gzip').to_numpy(dtype=dtype)
        x=read('raw/node-feat.csv.gz',np.float32)
        edge=read('raw/edge.csv.gz',np.int64).T.copy()
        y=read('raw/node-label.csv.gz',np.int64).reshape(-1)
        year=read('raw/node_year.csv.gz',np.int64).reshape(-1)
        split={k:read(f'split/time/{k}.csv.gz',np.int64).reshape(-1) for k in ['train','valid','test']}
    assert x.shape==(169343,128) and edge.shape==(2,1166243)
    assert y.min()==0 and y.max()==39
    joined=np.concatenate(list(split.values()));assert np.array_equal(np.sort(joined),np.arange(len(y)))
    assert np.all(year[split['train']]<=2017)
    assert np.all(year[split['valid']]==2018)
    assert np.all(year[split['test']]>=2019)
    audit={'archive_sha256':actual,'nodes':len(y),'directed_edges':edge.shape[1],'features':x.shape[1],'classes':40,'splits':{k:len(v) for k,v in split.items()},'year_ranges':{k:[int(year[v].min()),int(year[v].max())] for k,v in split.items()},'arrays_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in {'x':x,'edge':edge,'y':y,'year':year,**split}.items()}}
    return torch.from_numpy(x),torch.from_numpy(edge),torch.from_numpy(y),{k:torch.from_numpy(v) for k,v in split.items()},audit

# %% Full training and selected-state replay

@torch.no_grad()
def evaluate(model,x,adj,y,split):
    model.eval();pred=model(x,adj).argmax(1).cpu().numpy().reshape(-1,1)
    evaluator=Evaluator(name='ogbn-arxiv');labels=y.cpu().numpy().reshape(-1,1)
    scores={k:float(evaluator.eval({'y_true':labels[v.cpu().numpy()],'y_pred':pred[v.cpu().numpy()]})['acc']) for k,v in split.items()}
    return scores,pred.reshape(-1)

def train_run(x,adj,y,split,seed,epochs,output,device='cpu'):
    output=Path(output)
    if (output/'result.json').exists():raise FileExistsError('Use a fresh output directory; no silent resume')
    output.mkdir(parents=True,exist_ok=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    model=GCN().to(device);model.reset_parameters()
    x=x.to(device);adj=adj.to(device);y=y.to(device);split={k:v.to(device) for k,v in split.items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    history=[];best=-1.;state=None;start=time.perf_counter()
    for epoch in range(epochs):
        model.train();optimizer.zero_grad();loss=training_loss(model(x,adj),y,split['train']);loss.backward();optimizer.step()
        scores,_=evaluate(model,x,adj,y,split)
        history.append({'epoch':epoch+1,'loss':float(loss.detach()),**scores})
        # Strict improvement saves the first maximum, including BN buffers.
        if scores['valid']>best:
            best=scores['valid'];state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if (epoch+1)%50==0:print(json.dumps({'seed':seed,**history[-1],'elapsed':time.perf_counter()-start}),flush=True)
    chosen=selected_epoch([r['valid'] for r in history]);model.load_state_dict(state)
    scores,pred=evaluate(model,x,adj,y,split)
    assert scores=={k:history[chosen][k] for k in split}
    torch.save(state,output/'checkpoint.pt')
    np.savez_compressed(output/'predictions.npz',pred=pred,y=y.cpu().numpy(),**{k:v.cpu().numpy() for k,v in split.items()})
    result={'seed':seed,'epochs':epochs,'selected_epoch':chosen+1,'scores':scores,'seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters()),'history':history,'checkpoint_sha256':sha256(output/'checkpoint.pt'),'predictions_sha256':sha256(output/'predictions.npz'),'device':device,'torch':torch.__version__,'numpy':np.__version__,'status':'COMPLETE' if epochs==500 else 'TEACHING_ONLY'}
    (output/'result.json').write_text(json.dumps(result,indent=2));return result
