# %% Imports and execution contract
"""Cvitkovic Home Credit GCN port. Original MIT code is pinned in sources/l118.
Graph mechanics are visible PyTorch; real-data extraction uses the released prepared format.
A completed toy run is COURSE_ONLY. Five-fold paper replay is a separate explicit command.
"""
import copy
import hashlib
import json
import math
import pickle
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

# %% Task 1: two ordered reachability phases
def rdb_to_graph(n_nodes, edges, target):
    if not 0 <= target < n_nodes:
        raise ValueError('target outside graph')
    incoming=[[] for _ in range(n_nodes)];outgoing=[[] for _ in range(n_nodes)]
    for u,v in edges:
        if not (0 <= u < n_nodes and 0 <= v < n_nodes):
            raise ValueError('edge outside graph')
        outgoing[u].append(v);incoming[v].append(u)
    selected={target}
    for adjacency in [incoming,outgoing]:
        queue=list(selected)
        while queue:
            u=queue.pop()
            for v in adjacency[u]:
                if v not in selected:
                    selected.add(v);queue.append(v)
    return sorted(selected),[(u,v) for u,v in edges if u in selected and v in selected]

# %% Task 2: normalized message aggregation
def normalized_sum(x, edges):
    # The release collator already supplied reverse edges and self loops.
    src,dst=edges
    degree=torch.bincount(dst,minlength=len(x)).to(x.dtype).clamp_min(1)
    norm=degree.rsqrt()
    output=torch.zeros_like(x)
    output.index_add_(0,dst,x[src]*norm[src,None])
    return output*norm[:,None]

# %% Task 3: graph-local gated pooling
def attention_pool(values, gates, batch, n_graphs):
    gates=gates.reshape(-1)
    if (torch.bincount(batch,minlength=n_graphs)==0).any():
        raise ValueError('empty graph cannot be pooled')
    maximum=gates.new_full((n_graphs,),-torch.inf)
    maximum.scatter_reduce_(0,batch,gates,reduce='amax',include_self=True)
    exponent=torch.exp(gates-maximum[batch])
    denominator=gates.new_zeros(n_graphs).index_add_(0,batch,exponent)
    weights=exponent/denominator[batch]
    return values.new_zeros((n_graphs,values.shape[1])).index_add_(0,batch,weights[:,None]*values)

# %% Graph construction: stored edges differ from computation edges
def computation_edges(n_nodes, stored_edges):
    edges=list(stored_edges)
    edges=edges+[(v,u) for u,v in edges]+[(i,i) for i in range(n_nodes)]
    return torch.tensor(edges,dtype=torch.long).T.contiguous()


def undirected_hops(n_nodes,edges,target,hops=2,available=None,cutoff=None):
    """Home Credit's radius selection; optional time filter is a COURSE extension."""
    adjacency=[set() for _ in range(n_nodes)]
    for u,v in edges:adjacency[u].add(v);adjacency[v].add(u)
    selected={target};front={target}
    for _ in range(hops):
        candidates={v for u in front for v in adjacency[u]}
        front={v for v in candidates-selected if available is None or available[v] is None or available[v]<=cutoff}
        selected|=front
    return sorted(selected)

# %% Feature preprocessing: released metadata, no target column
def scalar_encode(values,center,scale):
    # Reproduce float32 input, missing flag, epsilon, and clipping in ScalarRescaleEnc.
    vals=np.array([np.nan if v is None else float(v) for v in values],dtype=np.float32)
    missing=np.isnan(vals)
    vals-=center;vals/=scale;vals+=1e-7
    vals[missing]=0
    return torch.from_numpy(np.column_stack([vals,missing.astype(np.float32)]).clip(-5,5))


def feature_schema(info):
    spec={}
    for table,features in info['node_types_and_features'].items():
        cards=[];n_cont=0
        for name,meta in features.items():
            if table+'.'+name==info['label_feature']:continue
            if meta['type']=='CATEGORICAL':cards.append(len(meta['sorted_values'])+1)
            elif meta['type']=='SCALAR':n_cont+=2
            else:raise ValueError('Home Credit port only covers its scalar/categorical columns')
        spec[table]=(cards,n_cont)
    return spec


def encode_features(raw,info):
    result={}
    for table,features in raw.items():
        cats=[];cont=[]
        for name,meta in info['node_types_and_features'][table].items():
            if table+'.'+name==info['label_feature']:continue
            values=features[name]
            if not values:continue
            if meta['type']=='CATEGORICAL':
                mapping={v:i+1 for i,v in enumerate(meta['sorted_values'])}
                cats.append(torch.tensor([mapping.get(v,0) for v in values],dtype=torch.long))
            elif meta['type']=='SCALAR':cont.append(scalar_encode(values,meta['RobustScaler_center_'],meta['RobustScaler_scale_']))
            else:raise ValueError(meta['type'])
        if cats or cont:
            n=len(cats[0]) if cats else len(cont[0])
            result[table]=(torch.stack(cats,dim=1) if cats else torch.empty(n,0,dtype=torch.long),torch.cat(cont,dim=1) if cont else torch.empty(n,0))
    return result

# %% Table-specific initializers: embeddings then d -> 4d -> hidden
class RowEncoder(nn.Module):
    def __init__(self,cards,n_cont,hidden=256,dropout=.5):
        super().__init__()
        self.embeddings=nn.ModuleList([nn.Embedding(c,min(32,c)) for c in cards])
        for embedding in self.embeddings:embedding.weight.data.clamp_(-2,2)
        self.dropout=nn.Dropout(dropout)
        width=sum(min(32,c) for c in cards)+n_cont
        self.layers=nn.Sequential(nn.Linear(width,4*width),nn.SELU(),nn.Dropout(dropout),nn.Linear(4*width,hidden),nn.SELU())
    def forward(self,cat,cont):
        parts=[self.dropout(e(cat[:,i])) for i,e in enumerate(self.embeddings)]
        if cont.shape[1]:parts.append(cont)
        return self.layers(torch.cat(parts,dim=1))

# %% Model architecture: one shared GCN, two readout branches, two logits
class CvitkovicGCN(nn.Module):
    def __init__(self,schema,type_ids,hidden=256,dropout=.5):
        super().__init__();self.type_ids=type_ids;self.hidden=hidden
        self.encoders=nn.ModuleDict({t:RowEncoder(cards,n,hidden,dropout) for t,(cards,n) in schema.items()})
        # Match original construction order: row encoders, readout, head, then GCN.
        self.gate=nn.Sequential(nn.Linear(hidden,hidden),nn.SELU(),nn.Linear(hidden,hidden),nn.SELU(),nn.Linear(hidden,1))
        self.value=nn.Sequential(nn.Linear(hidden,hidden),nn.SELU(),nn.Linear(hidden,hidden),nn.SELU())
        self.head=nn.Linear(hidden,2)
        self.weight=nn.Parameter(torch.empty(hidden,hidden));self.bias=nn.Parameter(torch.zeros(hidden))
        nn.init.xavier_uniform_(self.weight)
        self.dropout=nn.Dropout(dropout)
    def forward(self,features,node_types,edges,batch,n_graphs):
        h=self.weight.new_zeros((len(node_types),self.hidden))
        for table,(cat,cont) in features.items():h[node_types==self.type_ids[table]]=self.encoders[table](cat,cont)
        # Equal in/out widths: DGL 0.3.1 aggregates before its linear transform.
        h=self.dropout(F.selu(normalized_sum(h,edges)@self.weight+self.bias))
        return self.head(attention_pool(self.value(h),self.gate(h),batch,n_graphs))

# %% Prepared dataset and disjoint graph batching
class PreparedGraphs(torch.utils.data.Dataset):
    """Only load locally trusted author-produced pickle files."""
    def __init__(self,directory,ids):self.directory=Path(directory);self.ids=list(map(int,ids))
    def __len__(self):return len(self.ids)
    def __getitem__(self,index):
        identity=self.ids[index]
        with (self.directory/str(identity)).open('rb') as f:record=pickle.load(f)
        return identity,record


def collate_graphs(records,info):
    all_edges=[];types=[];batch=[];labels=[];ids=[];raw={t:{f:[] for f in fs if t+'.'+f!=info['label_feature']} for t,fs in info['node_types_and_features'].items()}
    for graph,(identity,(edges,node_types,edge_types,features,label)) in enumerate(records):
        offset=len(types);all_edges.append(computation_edges(len(node_types),edges)+offset)
        types.extend(node_types);batch.extend([graph]*len(node_types));labels.append(label);ids.append(identity)
        for t,fs in raw.items():
            for f,values in fs.items():values.extend(features[t][f])
    return (encode_features(raw,info),torch.tensor(types),torch.cat(all_edges,dim=1),torch.tensor(batch),len(records)),torch.tensor(labels,dtype=torch.long),np.array(ids)


def move_batch(inputs,device):
    features,types,edges,batch,n=inputs
    return ({t:(c.to(device),v.to(device)) for t,(c,v) in features.items()},types.to(device),edges.to(device),batch.to(device),n)

# %% Fold identities: same random seed, different held-out applicants
def released_folds(ids):
    ids=np.array(sorted(ids))
    for trainval,test in KFold(5,shuffle=True,random_state=14).split(ids):
        train,val=train_test_split(ids[trainval],test_size=.15,random_state=14)
        yield train,val,ids[test]

# %% Full trainer: validate before training, first best AUROC, patience 50
def evaluate(model,loader,device):
    model.eval();ys=[];ps=[];ids=[]
    with torch.no_grad():
        for inputs,labels,identities in loader:
            ps.append(model(*move_batch(inputs,device)).softmax(1)[:,1].cpu().numpy())
            ys.append(labels.numpy());ids.append(identities)
    return np.concatenate(ys),np.concatenate(ps),np.concatenate(ids)


def fit_fold(info,directory,fold,output,device='cpu',deadline=None,epochs=300,batch_size=1024,hidden=256,dropout=.5):
    """Defaults are released Home Credit settings. Overrides are COURSE/PILOT only."""
    from functools import partial
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(1234);np.random.seed(1234)
    train,val,test=list(released_folds(info['train_dp_ids']))[fold]
    collate=partial(collate_graphs,info=info)
    loaders=[torch.utils.data.DataLoader(PreparedGraphs(directory,ids),batch_size=batch_size,shuffle=(i==0),num_workers=0,collate_fn=collate) for i,ids in enumerate([train,val,test])]
    model=CvitkovicGCN(feature_schema(info),info['node_type_to_int'],hidden,dropout).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0)
    best=-1.;best_epoch=-1;history=[];started=time.monotonic()
    def guard():
        if deadline is not None and time.monotonic()>=deadline:
            (output/'incomplete.json').write_text(json.dumps({'status':'INCOMPLETE','reason':'aggregate runtime cutoff'}))
            raise TimeoutError('Aggregate runtime cutoff; no complete reproduction claimed')
    for epoch in range(epochs):
        guard();y,p,_=evaluate(model,loaders[1],device);score=float(roc_auc_score(y,p))
        history.append({'epoch':epoch,'validation_auroc':score})
        if score>best:
            best=score;best_epoch=epoch;torch.save(model.state_dict(),output/'best.pt')
        if epoch-best_epoch>=50:break
        model.train()
        for inputs,labels,_ in loaders[0]:
            guard();optimizer.zero_grad();loss=F.cross_entropy(model(*move_batch(inputs,device)),labels.to(device));loss.backward();optimizer.step()
    # Release's final post-loop validation is not considered for best_auroc selection.
    model.load_state_dict(torch.load(output/'best.pt',map_location=device,weights_only=True))
    guard();y,p,identities=evaluate(model,loaders[2],device)
    np.savez_compressed(output/'predictions.npz',y=y,p=p,ids=identities)
    np.savez_compressed(output/'splits.npz',train=train,val=val,test=test)
    result={'fold':fold,'test_auroc':float(roc_auc_score(y,p)),'best_val_auroc':best,'best_epoch':best_epoch,'history':history,'seconds':time.monotonic()-started,'n_test':len(y),'status':'COMPLETE_FOLD','environment':{'torch':torch.__version__,'numpy':np.__version__},'settings':{'epochs':epochs,'batch_size':batch_size,'hidden':hidden,'dropout':dropout,'seed':1234,'split_seed':14}}
    (output/'result.json').write_text(json.dumps(result,indent=2));return result
