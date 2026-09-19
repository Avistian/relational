"""L085: Li et al. Figure 2 reconstruction and separate Cora depth diagnostic.
GCN operations follow the MIT-licensed liqimai/gcn and tkipf/gcn sources.
No historical Figure 2 seeds or plotting driver were identified in the inspected release.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def support(adjacency):
    """Binary, symmetric, loop-free adjacency -> dense symmetric support and augmented degrees."""
    a=torch.as_tensor(adjacency,dtype=torch.float64)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not torch.equal(a,a.T):
        raise ValueError('Expected square undirected adjacency')
    if torch.any(a.diag()!=0) or torch.any((a!=0)&(a!=1)):
        raise ValueError('Use binary edges without self loops')
    augmented=a+torch.eye(len(a),dtype=a.dtype)
    d=augmented.sum(1)
    return augmented/d.sqrt()[:,None]/d.sqrt()[None,:],d

def smooth(s,h,steps):
    """TODO contract: return S^steps H without weights, activation, or in-place mutation."""
    if not isinstance(steps,int) or steps<0:raise ValueError('steps must be a nonnegative integer')
    for _ in range(steps):h=s@h
    return h

def collapse_metrics(h,degrees):
    """TODO contract: degree-corrected variance and mean off-diagonal cosine over nonzero rows.
    Variance is mean squared deviation of H/sqrt(d) from its column means.
    Cosine is undefined (None) when fewer than two rows are nonzero; report their count.
    """
    corrected=h/degrees.sqrt()[:,None]
    variance=float((corrected-corrected.mean(0)).square().mean())
    norms=h.norm(dim=1);nonzero=norms>0;u=h[nonzero]/norms[nonzero,None];n=len(u)
    cosine=float(((u.sum(0).square().sum()-n)/(n*(n-1))).clamp(-1,1)) if n>1 else None
    return {'degree_variance':variance,'mean_cosine':cosine,'nonzero_rows':n,'rms':float(h.square().mean().sqrt())}

def graph_layer(s,h,w):
    """TODO contract: [N,N] support, [N,F] states, [F,C] weights -> [N,C] states."""
    return s@(h@w)

class DeepGCN(nn.Module):
    """Glorot-uniform, bias-free GCN; width 16, hidden ReLU, linear final coordinates.
    Figure 2 uses eval mode, no optimization and two output coordinates.
    The separate Cora extension applies dropout .5 before every layer while training.
    """
    def __init__(self,features,outputs,depth,dropout=0.):
        super().__init__()
        if depth<1:raise ValueError('depth must be positive')
        dims=[features]+[16]*(depth-1)+[outputs]
        self.weights=nn.ParameterList([nn.Parameter(torch.empty(a,b)) for a,b in zip(dims,dims[1:])])
        for w in self.weights:nn.init.xavier_uniform_(w)
        self.dropout=dropout

    def forward(self,x,s,return_hidden=False):
        states=[]
        for i,w in enumerate(self.weights):
            x=graph_layer(s,F.dropout(x,self.dropout,training=self.training),w)
            if i<len(self.weights)-1:x=F.relu(x)
            states.append(x)
        return (x,states) if return_hidden else x

def karate_experiment(graph,seeds=100):
    """Full Figure 2 setup; depths 1..5; all 34 nodes; no training, selection, or data split.
    Labels only color points. Seeds 0..99 are local, never chosen by visual separation.
    """
    a=np.zeros((34,34))
    for i,j in graph['edges']:a[i,j]=a[j,i]=1
    s,d=support(a);s=s.float();d=d.float();x=torch.eye(34);runs=[]
    for seed in range(seeds):
        for depth in range(1,6):
            torch.manual_seed(seed*10+depth)
            model=DeepGCN(34,2,depth).eval()
            with torch.no_grad():h=model(x,s)
            runs.append({'seed':seed,'rng_seed':seed*10+depth,'depth':depth,'coordinates':h.tolist(),**collapse_metrics(h,d)})
    pure=[]
    for depth in [0,1,2,3,4,5,10,20,50,100,200]:
        pure.append({'depth':depth,**collapse_metrics(smooth(s.double(),x.double(),depth),d.double())})
    return {'experiment':'Li et al. 2018 Figure 2, full setup reconstruction','status':'EXECUTED_WITH_DISCLOSED_GAPS','runs':runs,'pure_propagation_extension':pure,'graph_sha256':hashlib.sha256(json.dumps(graph,sort_keys=True).encode()).hexdigest(),'historical_coordinate_parity':'INCOMPARABLE','gaps':['Historical random seeds and weights unavailable','Modern PyTorch random number generator','Hidden ReLU and linear output follow released GCN convention; Figure 2 plotting driver not identified','Binary edges; NetworkX interaction weights deliberately ignored'],'not_run':['Li et al. co-training/self-training classification tables','Original TensorFlow Figure 2 execution']}

def train_depth(data,depth,seed,max_epochs=200):
    """Separate extension: full fixed Cora split, width16, Adam .01, first-weight L2.
    Stop on current validation loss > previous-ten mean, using final weights as L082.
    No test access until optimization and stopping are finished.
    """
    torch.manual_seed(seed)
    x,s,y,train,valid,test=data;x=x.to_dense() if x.is_sparse else x
    model=DeepGCN(x.shape[1],int(y.max())+1,depth,.5)
    opt=torch.optim.Adam(model.parameters(),lr=.01,eps=1e-8)
    history=[]
    def objective(z,index):return F.cross_entropy(z[index],y[index])+.00025*model.weights[0].square().sum()
    for epoch in range(max_epochs):
        model.train();opt.zero_grad();objective(model(x,s),train).backward();opt.step()
        model.eval()
        with torch.no_grad():value=float(objective(model(x,s),valid))
        history.append(value)
        if epoch>10 and value>np.mean(history[-11:-1]):break
    with torch.no_grad():
        logits,states=model(x,s,True)
        # Diagonal S_ii=1/d_i because exactly one self-loop was added.
        degrees=1/s.to_dense().diag()
        representation=states[-2] if len(states)>1 else states[-1]
        diagnostics=collapse_metrics(representation,degrees)
        accuracy=float((logits[test].argmax(1)==y[test]).float().mean())
        train_accuracy=float((logits[train].argmax(1)==y[train]).float().mean())
    return {'depth':depth,'seed':seed,'epochs':epoch+1,'test_accuracy':accuracy,'train_accuracy':train_accuracy,'validation_loss':history,'representation':'last hidden ReLU, or output at depth1',**diagnostics}
