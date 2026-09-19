"""GAT Cora: full PyTorch port of PetarV-/GAT 5af87e7 (MIT).
Edge-only computation matches the dense released head without dropout;
RNG streams and floating point reductions are modern-framework deviations.
"""
from pathlib import Path
import copy
import hashlib
import json
import pickle
import time
import urllib.request
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F


def neighbor_softmax(scores, receiver, nodes):
    """Normalize edge scores separately for each RECEIVER; stable even for large logits."""
    maximum=scores.new_full((nodes,),-torch.inf)
    maximum.scatter_reduce_(0,receiver,scores.detach(),reduce='amax',include_self=True)
    weights=(scores-maximum[receiver]).exp()
    denominator=scores.new_zeros(nodes).index_add(0,receiver,weights)
    return weights/denominator[receiver]


def weighted_messages(z,edge,alpha):
    """edge[0] receives from edge[1]. Sum alpha_ij z_j; never average a second time."""
    return z.new_zeros(z.shape).index_add(0,edge[0],alpha[:,None]*z[edge[1]])


def merge_heads(heads,concat):
    """Hidden heads concatenate coordinates; prediction heads average class logits."""
    return torch.cat(heads,dim=-1) if concat else torch.stack(heads).mean(0)


def load_cora(root,manifest):
    """Verify every byte before unpickling pinned Planetoid data; preserve the fixed split."""
    root=Path(root)
    for record in manifest['data']:
        p=root/record['path'];p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():p.write_bytes(urllib.request.urlopen(record['url'],timeout=60).read())
        if hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Data hash mismatch: '+str(p))
    folder=root/'data/l078'
    def read(k):
        with (folder/('ind.cora.'+k)).open('rb') as f:return pickle.load(f,encoding='latin1')
    x,y,tx,ty,allx,ally,graph=[read(k) for k in ['x','y','tx','ty','allx','ally','graph']]
    reorder=np.loadtxt(folder/'ind.cora.test.index',dtype=int);ordered=np.sort(reorder)
    features=sp.vstack([allx,tx]).tolil();features[reorder]=features[ordered]
    labels=np.vstack([ally,ty]);labels[reorder]=labels[ordered]
    mass=np.asarray(features.sum(1)).ravel();features=(sp.diags(1/np.maximum(mass,1))@features).tocoo()
    xt=torch.sparse_coo_tensor(np.array([features.row,features.col]),features.data.astype('float32'),features.shape).coalesce()
    edges={(i,i) for i in range(len(labels))}
    for i,neighbors in graph.items():
        for j in neighbors:edges.update([(i,j),(j,i)])
    edge=torch.tensor(sorted(edges)).T.contiguous()
    return xt,edge,torch.tensor(labels.argmax(1)),torch.arange(len(y)),torch.arange(len(y),len(y)+500),torch.tensor(ordered)


class AttentionHead(nn.Module):
    """Released dense-head semantics: independent input dropout per head; scores use
    undropped projected features; independent projected-feature and coefficient dropout.
    Two scalar score biases and output bias are included, as in the TF release.
    """
    def __init__(self,features,width,dropout=.6):
        super().__init__();self.dropout=dropout
        self.w=nn.Parameter(torch.empty(features,width))
        self.a_receiver=nn.Parameter(torch.empty(width,1));self.a_sender=nn.Parameter(torch.empty(width,1))
        self.b_receiver=nn.Parameter(torch.zeros(()));self.b_sender=nn.Parameter(torch.zeros(()));self.bias=nn.Parameter(torch.zeros(width))
        for p in [self.w,self.a_receiver,self.a_sender]:nn.init.xavier_uniform_(p)

    def forward(self,x,edge,return_attention=False):
        if x.is_sparse:
            values=F.dropout(x.values(),self.dropout,self.training)
            xd=torch.sparse_coo_tensor(x.indices(),values,x.shape).coalesce()
            z=torch.sparse.mm(xd,self.w)
        else:z=F.dropout(x,self.dropout,self.training)@self.w
        q=(z@self.a_receiver).flatten()+self.b_receiver
        k=(z@self.a_sender).flatten()+self.b_sender
        scores=F.leaky_relu(q[edge[0]]+k[edge[1]],negative_slope=.2)
        alpha=neighbor_softmax(scores,edge[0],x.shape[0])
        coefficients=F.dropout(alpha,self.dropout,self.training)
        values=F.dropout(z,self.dropout,self.training)
        output=weighted_messages(values,edge,coefficients)+self.bias
        return (output,alpha) if return_attention else output


class GAT(nn.Module):
    """Cora: eight independent 8-coordinate heads, ELU, one 7-class output head."""
    def __init__(self,features=1433,classes=7):
        super().__init__();self.hidden=nn.ModuleList([AttentionHead(features,8) for _ in range(8)])
        self.output=nn.ModuleList([AttentionHead(64,classes)])

    def forward(self,x,edge):
        h=merge_heads([F.elu(head(x,edge)) for head in self.hidden],True)
        return merge_heads([head(h,edge) for head in self.output],False)


def train_cora(data,seed,max_epochs=100000,patience=100,return_model=False):
    """Full-batch Adam; released OR-reset / AND-save stopping, including equality.
    L2 covers ALL parameters: release compares full variable names against bare names,
    so its purported bias exclusion does not exclude scoped bias variables.
    Validation uses unregularized CE; test is inspected only after checkpoint restore.
    """
    torch.set_num_threads(1);torch.manual_seed(seed)
    x,edge,y,train,valid,test=data;model=GAT(x.shape[1],int(y.max())+1)
    optimizer=torch.optim.Adam(model.parameters(),lr=.005,betas=(.9,.999),eps=1e-8)
    best_loss=float('inf');best_acc=0.;waiting=0;trace=[];selected_epoch=None;state=None
    for epoch in range(max_epochs):
        model.train();optimizer.zero_grad()
        ce=F.cross_entropy(model(x,edge)[train],y[train])
        regularizer=.0005*.5*sum(p.square().sum() for p in model.parameters())
        (ce+regularizer).backward();optimizer.step();model.eval()
        with torch.no_grad():
            logits=model(x,edge);vl=float(F.cross_entropy(logits[valid],y[valid]));va=float((logits[valid].argmax(1)==y[valid]).double().mean())
        trace.append({'epoch':epoch+1,'train_ce':float(ce.detach()),'validation_ce':vl,'validation_accuracy':va})
        improves_acc=va>=best_acc;improves_loss=vl<=best_loss
        if improves_acc or improves_loss:
            if improves_acc and improves_loss:
                state=copy.deepcopy(model.state_dict());selected_epoch=epoch+1
            best_acc=max(va,best_acc);best_loss=min(vl,best_loss);waiting=0
        else:
            waiting+=1
            if waiting==patience:break
    if state is None:raise RuntimeError('No valid checkpoint')
    model.load_state_dict(state);model.eval()
    with torch.no_grad():accuracy=float((model(x,edge)[test].argmax(1)==y[test]).double().mean())
    result={'seed':seed,'epochs':epoch+1,'selected_epoch':selected_epoch,'stopped_by_patience':waiting==patience,'test_accuracy':accuracy,'trace':trace}
    return (result,model) if return_model else result


def summarize(runs):
    """Seeds measure optimization variation on ONE fixed dataset/split."""
    a=np.array([r['test_accuracy'] for r in runs])
    return {'experiment':'GAT Table 2 Cora; complete release-protocol PyTorch port','paper_mean':.830,'paper_sd':.007,'paper_runs':100,'runs':runs,'mean':float(a.mean()),'sample_sd':float(a.std(ddof=1)) if len(a)>1 else None,'historical_parity':'INCOMPARABLE','deviations':['PyTorch instead of TensorFlow 1; Adam numerical semantics and RNG differ','Edge-only softmax and sparse input dropout instead of dense operations; same deterministic equations, different reduction/random streams','Local seeds 0..99; historical seed identities unavailable'],'not_run':['Original TensorFlow parity','Citeseer, Pubmed, PPI and paper baselines']}
