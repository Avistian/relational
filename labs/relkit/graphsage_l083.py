"""Visible supervised GraphSAGE-mean port; see l083-reproduction.md for gaps."""
import copy
import hashlib
import json
import time
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler


def load_ppi(root, manifest):
    """Verify the complete Stanford archive before reading arrays; fit scaling on training nodes."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    archive = root / 'ppi.zip'
    if not archive.exists():
        urllib.request.urlretrieve(manifest['data']['url'], archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['data']['sha256'], 'PPI bytes differ'
    with zipfile.ZipFile(archive) as z:
        def member(suffix):
            matches = [n for n in z.namelist() if n.endswith(suffix)]
            assert len(matches) == 1, matches
            return z.open(matches[0])
        graph = json.load(member('-G.json'))
        ids = {int(k): int(v) for k, v in json.load(member('-id_map.json')).items()}
        labels = json.load(member('-class_map.json'))
        x = np.load(member('-feats.npy'), allow_pickle=False)
    nodes = [v for v in graph['nodes'] if 'val' in v and 'test' in v]
    n = len(ids); assert x.shape[0] == n
    masks = {key: np.zeros(n, dtype=bool) for key in ['train','valid','test']}
    y = np.zeros((n, len(next(iter(labels.values())))), dtype=np.float32)
    for v in nodes:
        i = ids[int(v['id'])]
        assert not (v['val'] and v['test'])
        masks['valid'][i] = v['val']; masks['test'][i] = v['test']
        masks['train'][i] = not (v['val'] or v['test'])
        y[i] = labels[str(v['id'])]
    # Legacy node-link JSON endpoints index graph['nodes'], rather than feature rows.
    adj = [set() for _ in range(n)]
    active = {int(v['id']) for v in nodes}
    for e in graph['links']:
        u = int(graph['nodes'][e['source']]['id']); v = int(graph['nodes'][e['target']]['id'])
        if u in active and v in active:
            a,b = ids[u],ids[v]; adj[a].add(b); adj[b].add(a)
    scaler = StandardScaler().fit(x[masks['train']])
    x = scaler.transform(x).astype(np.float32)
    # A zero sentinel is the released convention for empty neighbor lists.
    x = torch.from_numpy(np.vstack([x,np.zeros((1,x.shape[1]),dtype=np.float32)]))
    return {'x':x, 'y':torch.from_numpy(y), 'adj':[np.array(sorted(a),dtype=np.int64) for a in adj],
            'masks':masks, 'scaler_mean':scaler.mean_.tolist()}


def eligible_neighbors(adj, train_mask):
    """TODO: induced training graph. Both receiver and sender must be eligible."""
    return [a[train_mask[a]] if train_mask[i] else np.empty(0,dtype=np.int64) for i,a in enumerate(adj)]


def padded_adjacency(adj, max_degree, rng):
    """Release minibatch.py: cap long rows without replacement; resample short rows with replacement."""
    n = len(adj); table = np.full((n+1,max_degree),n,dtype=np.int64)
    for i, row in enumerate(adj):
        if len(row):
            table[i] = rng.choice(row,max_degree,replace=len(row)<max_degree) if len(row)!=max_degree else row
    return torch.from_numpy(table)


def sample_support(roots, table, fanouts, generator):
    """TODO: expand roots outward; one shared shuffled column order per hop matches the release."""
    samples = [roots]
    for size in fanouts:
        assert 0 < size <= table.shape[1]
        columns = torch.randperm(table.shape[1],generator=generator)[:size]
        samples.append(table[samples[-1]][:,columns].reshape(-1))
    return samples


def mean_concat(self_h, neighbor_h, w_self, w_neighbor):
    """TODO: release MeanAggregator(concat=True), before activation; neighbors have shape [B,S,d]."""
    return torch.cat([self_h @ w_self, neighbor_h.mean(dim=1) @ w_neighbor],dim=-1)


class MeanLayer(nn.Module):
    """Two distinct learned transforms; concatenation doubles the branch width."""
    def __init__(self, input_dim, branch_dim=128):
        super().__init__()
        self.w_self = nn.Parameter(torch.empty(input_dim,branch_dim))
        self.w_neighbor = nn.Parameter(torch.empty(input_dim,branch_dim))
        nn.init.xavier_uniform_(self.w_self); nn.init.xavier_uniform_(self.w_neighbor)
    def forward(self, h, neighbors):
        return mean_concat(h,neighbors,self.w_self,self.w_neighbor)


class GraphSAGE(nn.Module):
    """Full two-layer released mean encoder + final L2 normalization + multilabel linear head."""
    def __init__(self, input_dim, classes, branch_dim=128):
        super().__init__()
        self.layers = nn.ModuleList([MeanLayer(input_dim,branch_dim),MeanLayer(2*branch_dim,branch_dim)])
        self.head = nn.Linear(2*branch_dim,classes)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x, samples, fanouts=(10,25)):
        hidden = [x[s] for s in samples]
        for layer_index,layer in enumerate(self.layers):
            new_hidden = []
            for hop in range(len(hidden)-1):
                neighbors = hidden[hop+1].reshape(len(hidden[hop]),fanouts[hop],-1)
                h = layer(hidden[hop],neighbors)
                new_hidden.append(F.relu(h) if layer_index == 0 else h)
            hidden = new_hidden
        # TF l2_normalize uses sqrt(max(sum(x*x),epsilon)), with epsilon=1e-12.
        h = hidden[0] / hidden[0].square().sum(-1,keepdim=True).clamp_min(1e-12).sqrt()
        return self.head(h)


def micro_f1(logits, labels):
    """Pool label decisions across nodes/classes. Logit >0 equals sigmoid >0.5."""
    pred = logits > 0; true = labels.bool()
    tp = (pred & true).sum().item(); fp=(pred & ~true).sum().item(); fn=(~pred & true).sum().item()
    return 2*tp/max(2*tp+fp+fn,1)


def evaluate(model, data, table, nodes, fanouts, batch_size, seed):
    """Inference uses held-out features/edges, never fitting or adapting weights."""
    gen=torch.Generator().manual_seed(seed); model.eval(); out=[]
    with torch.no_grad():
        for start in range(0,len(nodes),batch_size):
            roots=torch.as_tensor(nodes[start:start+batch_size])
            out.append(model(data['x'],sample_support(roots,table,fanouts,gen),fanouts))
    logits=torch.cat(out); labels=data['y'][nodes]
    return {'micro_f1':micro_f1(logits,labels),'loss':float(F.binary_cross_entropy_with_logits(logits,labels))}


def fit_candidate(data, seed, learning_rate, config, tables=None):
    """Full schedule; no best-epoch selection. Validation selects only the learning rate."""
    torch.manual_seed(seed); rng=np.random.RandomState(seed)
    induced=eligible_neighbors(data['adj'],data['masks']['train'])
    if tables is None:
        tables=(padded_adjacency(induced,config['max_degree'],rng),padded_adjacency(data['adj'],config['max_degree'],rng))
    train_table,inference_table=tables
    roots=np.array([i for i in np.flatnonzero(data['masks']['train']) if len(induced[i])],dtype=np.int64)
    model=GraphSAGE(data['x'].shape[1],data['y'].shape[1],config['branch_dim'])
    opt=torch.optim.Adam(model.parameters(),lr=learning_rate,eps=1e-8)
    gen=torch.Generator().manual_seed(seed+1000); trace=[]; started=time.time()
    for epoch in range(config['epochs']):
        order=rng.permutation(roots); total=0.; count=0; model.train()
        for start in range(0,len(order),config['batch_size']):
            batch=torch.from_numpy(order[start:start+config['batch_size']])
            samples=sample_support(batch,train_table,config['fanouts'],gen)
            logits=model(data['x'],samples,config['fanouts'])
            loss=F.binary_cross_entropy_with_logits(logits,data['y'][batch])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_value_(model.parameters(),5.); opt.step()
            total+=float(loss.detach())*len(batch);count+=len(batch)
        trace.append({'epoch':epoch+1,'train_loss':total/count})
    valid=evaluate(model,data,inference_table,np.flatnonzero(data['masks']['valid']),config['fanouts'],config['batch_size'],seed+2000)
    return model,{'lr':learning_rate,'validation':valid,'trace':trace,'seconds':time.time()-started},inference_table


def experiment(data, seeds, config, output=None):
    """Three-rate validation sweep, final-weight evaluation, frozen test once per selected model."""
    result={'scope':'full PPI release-protocol port' if config['epochs']==10 and config['branch_dim']==128 else 'teaching run',
            'paper_target':.598,'status':'INCOMPARABLE','config':config,'runs':[],
            'versions':{'torch':torch.__version__,'numpy':np.__version__},
            'data_shape':list(data['x'].shape),'split_sizes':{k:int(v.sum()) for k,v in data['masks'].items()}}
    for seed in seeds:
        candidates=[]; best=None
        for lr in config['learning_rates']:
            model,record,table=fit_candidate(data,seed,lr,config)
            candidates.append(record)
            if best is None or record['validation']['micro_f1']>best[0]:
                best=(record['validation']['micro_f1'],copy.deepcopy(model.state_dict()),lr,table)
            print(f'seed={seed} lr={lr} validation={record["validation"]["micro_f1"]:.4f}',flush=True)
        model.load_state_dict(best[1])
        test=evaluate(model,data,best[3],np.flatnonzero(data['masks']['test']),config['fanouts'],config['batch_size'],seed+3000)
        result['runs'].append({'seed':seed,'candidates':candidates,'selected_lr':best[2],'test':test})
        scores=[r['test']['micro_f1'] for r in result['runs']]
        result['mean']=float(np.mean(scores));result['sample_sd']=float(np.std(scores,ddof=1)) if len(scores)>1 else None
        if output:Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result


PAPER_CONFIG={'epochs':10,'branch_dim':128,'batch_size':512,'max_degree':128,'fanouts':[10,25],'learning_rates':[.01,.001,.0001]}
