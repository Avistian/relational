"""L088: complete MUTAG GIN-0 release reconstruction; CPU deterministic execution.

Port of powerful-gnns at 9a2ce8ac3e99278307093a464a95caf0fb04b602.
Graph readout ablations are separate from the paper protocol.
"""
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold

COMMIT = '9a2ce8ac3e99278307093a464a95caf0fb04b602'
DATA_SHA = '5897dae243f6c773aab54ec99e86551c3b1e8601acef254714073042c632d30e'
ARCHIVE_SHA = '644254dd26f66ad41f47f55aed2d31c20e8884c63629e6bf605e0f82aad76e99'

def load_mutag(folder='l088-data'):
    """Released 188 graphs, encounter-order label/tag encoding; edge labels unused."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'MUTAG.txt'
    if not path.exists():
        url = f'https://raw.githubusercontent.com/weihua916/powerful-gnns/{COMMIT}/dataset.zip'
        raw = urllib.request.urlopen(url, timeout=120).read()
        if hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA:
            raise ValueError('Source archive checksum mismatch')
        path.write_bytes(zipfile.ZipFile(io.BytesIO(raw)).read('dataset/MUTAG/MUTAG.txt'))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA:
        raise ValueError('MUTAG checksum mismatch')
    lines = iter(raw.decode().splitlines()); count = int(next(lines))
    labels, tags, graphs = {}, {}, []
    for graph_id in range(count):
        n, label = map(int, next(lines).split()); labels.setdefault(label, len(labels))
        node_tags, edges = [], set()
        for v in range(n):
            row = list(map(int, next(lines).split())); tag, degree = row[:2]
            assert len(row) == degree + 2
            tags.setdefault(tag, len(tags)); node_tags.append(tags[tag])
            for u in row[2:]:
                assert 0 <= u < n and u != v
                edges.add((min(u,v), max(u,v)))
        pairs = sorted(edges); pairs += [(v,u) for u,v in pairs]
        graphs.append({'id':graph_id,'tags':node_tags,'edges':torch.tensor(pairs).T.contiguous(),
                       'y':labels[label]})
    for g in graphs:
        g['x'] = F.one_hot(torch.tensor(g.pop('tags')), num_classes=len(tags)).float()
    assert len(graphs)==188 and len(tags)==7 and len(labels)==2
    return graphs

def neighbor_sum(h, edge_index):
    """Sum incoming messages; edge_index[0] sends to edge_index[1]. No self edges."""
    result = torch.zeros_like(h)
    result.index_add_(0, edge_index[1], h[edge_index[0]])
    return result

def graph_readout(h, batch, num_graphs, mode='sum'):
    """Reduce node rows independently for each graph. Max is a teaching extension."""
    result = h.new_zeros((num_graphs,h.shape[1]))
    if mode == 'max':
        result.fill_(-torch.inf)
        return result.scatter_reduce(0,batch[:,None].expand_as(h),h,reduce='amax',include_self=True)
    result.index_add_(0,batch,h)
    if mode == 'mean':
        result = result / torch.bincount(batch,minlength=num_graphs)[:,None]
    elif mode != 'sum':
        raise ValueError(mode)
    return result

def select_epoch(fold_curves):
    """One common epoch maximizes equally weighted mean fold accuracy; earliest tie."""
    curves = np.asarray(fold_curves,dtype=float)
    if curves.ndim != 2 or not np.isfinite(curves).all():
        raise ValueError('Expected finite fold-by-epoch curves')
    return int(curves.mean(axis=0).argmax())

def collate(graphs):
    """Disjoint union: offset node IDs, retain one graph ID per node, graph labels."""
    xs,edges,batch,labels=[],[],[],[];offset=0
    for i,g in enumerate(graphs):
        xs.append(g['x']);edges.append(g['edges']+offset)
        batch.append(torch.full((len(g['x']),),i,dtype=torch.long))
        labels.append(g['y']);offset+=len(g['x'])
    return torch.cat(xs),torch.cat(edges,dim=1),torch.cat(batch),torch.tensor(labels)

class MLP(nn.Module):
    """Two affine layers, internal BatchNorm and ReLU, linear output."""
    def __init__(self,input_dim,hidden_dim):
        super().__init__()
        self.linears=nn.ModuleList([nn.Linear(input_dim,hidden_dim),nn.Linear(hidden_dim,hidden_dim)])
        self.batch_norms=nn.ModuleList([nn.BatchNorm1d(hidden_dim)])
    def forward(self,x):
        return self.linears[1](F.relu(self.batch_norms[0](self.linears[0](x))))

class GIN(nn.Module):
    """Four message updates plus input readout, matching the release's five layers.

    Each update has its own MLP and outer BatchNorm. Nodes/graphs share these
    weights. Linear prediction heads differ by depth; dropout acts on logits.
    """
    def __init__(self,input_dim=7,hidden_dim=16,classes=2,dropout=0.,learn_eps=False,readout='sum'):
        super().__init__();self.dropout=dropout;self.learn_eps=learn_eps;self.readout=readout
        self.eps=nn.Parameter(torch.zeros(4)) # unused, as in release, for GIN-0
        self.mlps=nn.ModuleList();self.batch_norms=nn.ModuleList()
        for layer in range(4):
            self.mlps.append(MLP(input_dim if layer==0 else hidden_dim,hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.linears_prediction=nn.ModuleList([nn.Linear(input_dim if k==0 else hidden_dim,classes) for k in range(5)])
    def forward(self,packed):
        h,edges,batch,labels=packed;states=[h]
        for k in range(4):
            eps=self.eps[k] if self.learn_eps else 0.
            h=neighbor_sum(h,edges)+(1+eps)*h
            h=F.relu(self.batch_norms[k](self.mlps[k](h)));states.append(h)
        score=0
        for k,h in enumerate(states):
            pooled=graph_readout(h,batch,len(labels),self.readout)
            score=score+F.dropout(self.linears_prediction[k](pooled),self.dropout,training=self.training)
        return score

def folds_for(graphs,split_seed=0):
    y=[g['y'] for g in graphs]
    return list(StratifiedKFold(n_splits=10,shuffle=True,random_state=split_seed).split(np.zeros(len(y)),y))

@torch.no_grad()
def evaluate(model,graphs):
    model.eval();out=[]
    for start in range(0,len(graphs),64):out.append(model(collate(graphs[start:start+64])))
    logits=torch.cat(out);labels=torch.tensor([g['y'] for g in graphs])
    return float((logits.argmax(1)==labels).float().mean()),logits

def train_fold(graphs,fold=0,hidden=16,batch_size=32,dropout=0.,epochs=350,iters=50,
               seed=0,split_seed=0,readout='sum'):
    """Full-size release recipe. An epoch means 50 random subset updates, not a pass.

    PyTorch 1.0 StepLR behavior is explicit: epochs 1..50 use .01, 51..100 .005.
    Validation never changes updates. Selection is performed after all folds.
    """
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    train_ids,val_ids=folds_for(graphs,split_seed)[fold]
    train=[graphs[i] for i in train_ids];valid=[graphs[i] for i in val_ids]
    model=GIN(graphs[0]['x'].shape[1],hidden,2,dropout,False,readout)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    curves=[];logits_by_epoch=[]
    for epoch in range(1,epochs+1):
        lr=.01 * .5**((epoch-1)//50)
        for group in optimizer.param_groups:group['lr']=lr
        model.train();loss_sum=0.
        for _ in range(iters):
            ids=np.random.permutation(len(train))[:batch_size]
            packed=collate([train[i] for i in ids]);logits=model(packed)
            loss=F.cross_entropy(logits,packed[3]);optimizer.zero_grad();loss.backward();optimizer.step()
            loss_sum+=float(loss.detach())
        acc,logits=evaluate(model,valid);train_acc,_=evaluate(model,train)
        # Exact integer ratio avoids rounding mean(float32) during model selection.
        acc=float((logits.argmax(1)==torch.tensor([g['y'] for g in valid])).sum())/len(valid)
        curves.append({'epoch':epoch,'lr':lr,'loss':loss_sum/iters,'train_acc':train_acc,'val_acc':acc})
        logits_by_epoch.append(logits.numpy())
    record={'config':{'hidden':hidden,'batch_size':batch_size,'dropout':dropout,'epochs':epochs,
            'iters':iters,'seed':seed,'split_seed':split_seed,'readout':readout},'fold':fold,
            'train_ids':train_ids.tolist(),'val_ids':val_ids.tolist(),'curves':curves}
    return record,np.stack(logits_by_epoch)

def paper_grid():
    return [{'hidden':h,'batch_size':b,'dropout':d} for h in [16,32] for b in [32,128] for d in [0.,.5]]

def summarize_grid(records):
    """Fail closed on incomplete grid; select one config and one common epoch."""
    candidates=[]
    for config in paper_grid():
        rows=sorted([r for r in records if all(r['config'][k]==v for k,v in config.items())],key=lambda r:r['fold'])
        assert [r['fold'] for r in rows]==list(range(10))
        assert all(r['config']['epochs']==350 and r['config']['iters']==50 and r['config']['readout']=='sum' for r in rows)
        curves=np.array([[c['val_acc'] for c in r['curves']] for r in rows]);epoch=select_epoch(curves)
        values=curves[:,epoch]
        candidates.append({'config':config,'epoch':epoch+1,'mean':float(values.mean()),
            'fold_values':values.tolist(),'population_sd':float(values.std(ddof=0)),
            'sample_sd':float(values.std(ddof=1))})
    winner=max(candidates,key=lambda r:r['mean'])
    return {'target':'Xu et al. 2019 Table 1 MUTAG GIN-0','paper_mean':.894,'paper_sd':.056,
            'status':'FULL_GRID_EXECUTED','historical_parity':'INCOMPARABLE',
            'reason':'Modern runtime/splitter and edge reduction order; historical split IDs and selected config unavailable.',
            'selection':'Maximum mean validation curve, common epoch; first config and earliest epoch break ties. No independent test set.',
            'candidates':candidates,'selected':winner,'fold_runs':len(records),'optimizer_updates':80*350*50}
