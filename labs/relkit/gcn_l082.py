"""L082: complete Cora GCN release-protocol port, shared provenance with L078.
GCN reference: tkipf/gcn 39a4089fe72ad9f055ed6fdb9746abdcfebc4d81 (MIT).
The modern PyTorch RNG/Adam are not bitwise TensorFlow-1 equivalents.
"""
from pathlib import Path
import hashlib
import json
import pickle
import urllib.request
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F


def as_sparse(a):
    a=sp.coo_matrix(a)
    return torch.sparse_coo_tensor(np.array([a.row,a.col]),a.data.astype('float32'),a.shape).coalesce()



def normalized_support(a):
    """Binary undirected loop-free adjacency -> sparse D^-1/2 (A+I) D^-1/2.
    Add exactly one self-loop per node BEFORE computing degrees; retain isolated nodes.
    """
    a=sp.csr_matrix(a,dtype=float)
    if a.shape[0]!=a.shape[1] or (a-a.T).nnz or np.any(a.diagonal()):
        raise ValueError('Expected square undirected adjacency with no self-loops')
    if np.any(a.data!=1):raise ValueError('Expected binary adjacency')
    t=a+sp.eye(a.shape[0]);d=np.asarray(t.sum(1)).ravel()**-.5
    return as_sparse(sp.diags(d)@t@sp.diags(d))


def propagate(s,h,w):
    """S [N,N] times H [N,F] times W [F,C] -> [N,C]; S is sparse."""
    hw=torch.sparse.mm(h,w) if h.is_sparse else h@w
    return torch.sparse.mm(s,hw)


def masked_objective(logits,y,index,w0):
    """Mean cross-entropy on selected nodes plus release L2 on FIRST weights only."""
    return F.cross_entropy(logits[index],y[index])+.0005*.5*w0.square().sum()

def load_cora(root=None):
    """Verified pinned pickles only. Download missing files; reject changed bytes."""
    if root is None:
        root=Path.cwd()
        if (root/'labs').is_dir():root=root/'labs'
        if not (root/'_sources_l078.json').exists():
            root=Path(__file__).resolve().parents[1]
    root=Path(root);manifest=json.loads((root/'_sources_l078.json').read_text())
    for record in manifest['files']:
        if not record['path'].startswith('data/'):continue
        p=root/record['path'];p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            url='https://raw.githubusercontent.com/tkipf/gcn/'+manifest['revision']+'/'+record['upstream']
            p.write_bytes(urllib.request.urlopen(url,timeout=60).read())
        if hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Pinned data hash mismatch: '+str(p))
    folder=root/'data/l078'
    def read(name):
        with (folder/('ind.cora.'+name)).open('rb') as f:return pickle.load(f,encoding='latin1')
    x,y,tx,ty,allx,ally,graph=[read(k) for k in ['x','y','tx','ty','allx','ally','graph']]
    reorder=np.loadtxt(folder/'ind.cora.test.index',dtype=int);ordered=np.sort(reorder)
    features=sp.vstack((allx,tx)).tolil();features[reorder]=features[ordered]
    labels=np.vstack((ally,ty));labels[reorder]=labels[ordered]
    rows=[];cols=[]
    for i,neighbors in graph.items():
        for j in neighbors:rows.extend([i,j]);cols.extend([j,i])
    a=sp.coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(labels),len(labels))).tocsr();a.data[:]=1
    s=normalized_support(a)
    mass=np.asarray(features.sum(1)).ravel();features=sp.diags(1/np.maximum(mass,1))@features
    return as_sparse(features),s,torch.tensor(labels.argmax(1)),torch.arange(len(y)),torch.arange(len(y),len(y)+500),torch.tensor(ordered)


class GCN(nn.Module):
    """Two bias-free layers, Glorot weights, sparse feature dropout and hidden dropout."""
    def __init__(self, features, classes):
        super().__init__()
        self.w0=nn.Parameter(torch.empty(features,16));self.w1=nn.Parameter(torch.empty(16,classes))
        nn.init.xavier_uniform_(self.w0);nn.init.xavier_uniform_(self.w1)

    def forward(self,x,s):
        if self.training:
            v=F.dropout(x.values(),p=.5,training=True)
            x=torch.sparse_coo_tensor(x.indices(),v,x.shape).coalesce()
        h=torch.relu(propagate(s,x,self.w0))
        h=F.dropout(h,p=.5,training=self.training)
        return propagate(s,h,self.w1)


def train_cora(data,seed):
    """Full released schedule: current validation loss vs previous-10 mean, last weights."""
    torch.manual_seed(seed)
    x,s,y,train,valid,test=data
    model=GCN(x.shape[1],int(y.max())+1)
    opt=torch.optim.Adam(model.parameters(),lr=.01,betas=(.9,.999),eps=1e-8)
    history=[]
    def loss(logits,idx):
        return masked_objective(logits,y,idx,model.w0)
    for epoch in range(200):
        model.train();opt.zero_grad();loss(model(x,s),train).backward();opt.step()
        model.eval()
        with torch.no_grad():vl=float(loss(model(x,s),valid))
        history.append(vl)
        if epoch>10 and vl>np.mean(history[-11:-1]):break
    # Test labels are consumed once, after the validation stopping decision.
    with torch.no_grad():
        logits=model(x,s);accuracy=float((logits[test].argmax(1)==y[test]).float().mean())
    return {'seed':seed,'epochs':epoch+1,'test_accuracy':accuracy,'validation_loss':history}


def run_cora(seeds=100,root=None):
    torch.set_num_threads(1)
    data=load_cora(root);runs=[]
    for seed in range(seeds):
        r=train_cora(data,seed);runs.append(r)
        print(f"seed {seed}: {r['test_accuracy']:.4f}, {r['epochs']} epochs",flush=True)
    a=np.array([r['test_accuracy'] for r in runs])
    return {'experiment':'Kipf-Welling Table 2 Cora fixed split; PyTorch release-protocol port','paper_target':.815,'runs':runs,'mean':float(a.mean()),'sample_sd':float(a.std(ddof=1)) if seeds>1 else None,'se':float(a.std(ddof=1)/np.sqrt(seeds)) if seeds>1 else None,'torch':torch.__version__,'numpy':np.__version__,'deviations':['PyTorch rather than TensorFlow 1; RNG and Adam numerical semantics differ','Seeds 0..99 declared locally; original 100 seed identities unavailable','Release early-stop mean rule and last weights used; paper prose describes consecutive non-improvement'],'not_run':['Other Table 2 datasets and baselines','Original TensorFlow training parity']}
