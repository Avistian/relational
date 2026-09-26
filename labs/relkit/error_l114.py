# %% Imports and frozen protocol
"""L114: visible OGB MLP and retrospective error analysis.
MLP architecture mirrors the MIT-licensed OGB release in sources/l114/.
"""
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TARGETS={'valid':57.65,'test':55.50}
TOLERANCE_PP=.5

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

# %% Complete feature-only architecture
class MLP(nn.Module):
    """Released three-layer MLP; BN sees training rows during fitting."""
    def __init__(self,in_channels=128,hidden_channels=256,out_channels=40,dropout=.5):
        super().__init__()
        self.lins=nn.ModuleList([nn.Linear(in_channels,hidden_channels),nn.Linear(hidden_channels,hidden_channels),nn.Linear(hidden_channels,out_channels)])
        self.bns=nn.ModuleList([nn.BatchNorm1d(hidden_channels),nn.BatchNorm1d(hidden_channels)])
        self.dropout=dropout
    def reset_parameters(self):
        for lin in self.lins:lin.reset_parameters()
        for bn in self.bns:bn.reset_parameters()
    def forward(self,x):
        for lin,bn in zip(self.lins[:-1],self.bns):
            x=F.dropout(F.relu(bn(lin(x))),p=self.dropout,training=self.training)
        return self.lins[-1](x).log_softmax(dim=-1)

# %% Full released training schedule and checkpoint selection
@torch.no_grad()
def mlp_evaluate(model,x,y,split):
    model.eval();pred=model(x).argmax(1)
    scores={k:float((pred[v]==y[v]).float().sum().item()/len(v)) for k,v in split.items()}
    return scores,pred.cpu().numpy()

def train_mlp(x,y,split,seed,epochs,output,device='cpu'):
    """No silent resume; first validation maximum; save BN running buffers."""
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use a fresh empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    (output/'started.json').write_text(json.dumps({'seed':seed,'epochs':epochs}))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    model=MLP().to(device);model.reset_parameters()
    x=x.to(device);y=y.to(device);split={k:v.to(device) for k,v in split.items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    history=[];best=-1.;state=None;start=time.perf_counter()
    for epoch in range(epochs):
        model.train();optimizer.zero_grad()
        # Forward only train rows: this determines the BatchNorm population.
        loss=F.nll_loss(model(x[split['train']]),y[split['train']])
        loss.backward();optimizer.step()
        scores,_=mlp_evaluate(model,x,y,split)
        history.append({'epoch':epoch+1,'loss':float(loss.detach()),**scores})
        if scores['valid']>best:
            best=scores['valid'];state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if (epoch+1)%100==0:print(json.dumps({'seed':seed,**history[-1]}),flush=True)
    chosen=int(np.argmax([h['valid'] for h in history]));model.load_state_dict(state)
    scores,pred=mlp_evaluate(model,x,y,split)
    assert scores=={k:history[chosen][k] for k in split}
    torch.save(state,output/'checkpoint.pt')
    np.savez_compressed(output/'predictions.npz',pred=pred,y=y.cpu().numpy(),**{k:v.cpu().numpy() for k,v in split.items()})
    result={'seed':seed,'epochs':epochs,'selected_epoch':chosen+1,'scores':scores,'history':history,'seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':file_hash(output/'checkpoint.pt'),'predictions_sha256':file_hash(output/'predictions.npz'),'device':device,'torch':torch.__version__,'numpy':np.__version__,'status':'COMPLETE' if epochs==500 else 'TEACHING_ONLY'}
    (output/'result.json').write_text(json.dumps(result,indent=2));return result

# %% Task 1: define a neighbor before measuring one

def neighborhood_properties(edge, labels, train_ids):
    """Unique undirected neighbors, no self loops. All-label homophily is retrospective.

    train_neighbor_fraction uses membership only; it never compares target labels.
    Isolates return NaN for both fractions, not a made-up zero.
    """
    labels=np.asarray(labels);n=len(labels);e=np.asarray(edge,dtype=np.int64)
    src=np.concatenate([e[0],e[1]]);dst=np.concatenate([e[1],e[0]])
    keep=src!=dst;keys=np.unique(src[keep]*n+dst[keep]);row=keys//n;col=keys%n
    degree=np.bincount(row,minlength=n)
    same=np.bincount(row,weights=(labels[row]==labels[col]),minlength=n)
    is_train=np.zeros(n,dtype=bool);is_train[train_ids]=True
    known=np.bincount(row,weights=is_train[col],minlength=n)
    hom=np.full(n,np.nan);coverage=np.full(n,np.nan)
    np.divide(same,degree,out=hom,where=degree>0);np.divide(known,degree,out=coverage,where=degree>0)
    return {'degree':degree,'homophily':hom,'train_neighbor_fraction':coverage}

# %% Task 2: compare the same questions, retain the denominator

def slice_metrics(labels, gcn_predictions, mlp_predictions, ids):
    """Inputs predictions[seed,node]. Discordance: both, GCN-only, MLP-only, neither."""
    ids=np.asarray(ids,dtype=np.int64);n=len(ids)
    if not n:return {'n':0,'gcn':None,'mlp':None,'delta_pp':None,'mean_delta_pp':None,'discordance':None}
    a=np.asarray(gcn_predictions)[:,ids]==np.asarray(labels)[ids]
    b=np.asarray(mlp_predictions)[:,ids]==np.asarray(labels)[ids]
    ga=a.mean(1);mb=b.mean(1);delta=(ga-mb)*100
    discord=np.stack([(a&b).sum(1),(a&~b).sum(1),(~a&b).sum(1),(~a&~b).sum(1)],axis=1)
    return {'n':n,'gcn':ga.tolist(),'mlp':mb.tolist(),'delta_pp':delta.tolist(),'mean_delta_pp':float(delta.mean()),'discordance':discord.tolist()}

# %% Task 3: validation nominates, test describes

def choose_failure(rows, minimum_count=200):
    """Caller supplies validation rows only; stable tie-break by family and slice."""
    eligible=[r for r in rows if r['n']>=minimum_count and r['mean_delta_pp'] is not None]
    if not eligible:return None
    return min(eligible,key=lambda r:(r['mean_delta_pp'],r['family'],r['slice']))

# %% Predeclared slice definitions, identical across models

def slice_masks(properties,labels,year):
    d=properties['degree'];h=properties['homophily'];m={}
    for lo,hi,name in [(0,1,'0'),(1,3,'1–2'),(3,6,'3–5'),(6,11,'6–10'),(11,21,'11–20'),(21,51,'21–50'),(51,np.inf,'51+')]:
        m[('degree',name)]=(d>=lo)&(d<hi)
    m[('homophily','undefined')]=np.isnan(h)
    for lo,hi,name in [(0,.25,'0–<.25'),(.25,.5,'.25–<.50'),(.5,.75,'.50–<.75'),(.75,1.01,'.75–1')]:
        m[('homophily',name)]=(h>=lo)&(h<hi)
    for c in range(40):m[('class',str(c))]=labels==c
    for t in np.unique(year):m[('year',str(t))]=year==t
    return m
