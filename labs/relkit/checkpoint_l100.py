"""L100 checkpoint. L099 visible operators retained; native sampled training added.
Course experiment, not a historical paper benchmark.
"""
import copy, hashlib, json, math, platform, time, urllib.request

from pathlib import Path

import numpy as np

import scipy.io as sio

import torch

from torch import nn

from torch.nn import functional as F

ROUTES = [(0,1),(1,0),(0,2),(2,0)]

ARMS = ['rgcn','hgt','hgt_uniform','mlp']

DATA_URL = 'https://data.dgl.ai/dataset/ACM.mat'

DATA_SHA256 = '0ccd838e545e8f16e3dc84356da2f51dfd2290c32e37a784e374dd510c76578d'

LEARNING_RATES = [.003,.01]

SEEDS = [0,1,2]

EPOCHS = 40

def relation_mean(x, edge, receivers):
    """ mean incoming source rows per receiver; empty neighborhoods return zero."""
    src,dst=edge
    out=x.new_zeros((receivers,x.shape[1])).index_add(0,dst,x[src])
    degree=torch.bincount(dst,minlength=receivers).clamp_min(1)
    return out/degree[:,None]

def receiver_softmax(scores, dst, receivers):
    """ stable E×heads softmax over ALL incoming relations per receiver/head."""
    ix=dst[:,None].expand_as(scores)
    maximum=scores.new_full((receivers,scores.shape[1]),-torch.inf)
    maximum.scatter_reduce_(0,ix,scores.detach(),reduce='amax',include_self=True)
    weights=(scores-maximum[dst]).exp()
    denominator=scores.new_zeros(maximum.shape).scatter_add(0,ix,weights)
    return weights/denominator[dst].clamp_min(torch.finfo(scores.dtype).tiny)

def choose_config(records, arm):
    """ minimize mean best validation CE across seeds; ties use smaller LR."""
    rates=sorted({r['lr'] for r in records if r['arm']==arm})
    return min(rates,key=lambda lr:(np.mean([r['best_val_ce'] for r in records if r['arm']==arm and r['lr']==lr]),lr))

def load_graph(root):
    """Full DGL ACM conference-filtered graph; fixed course split; no conference edges."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True);path=root/'ACM.mat'
    if not path.exists():
        with urllib.request.urlopen(DATA_URL,timeout=120) as r:payload=r.read()
        if hashlib.sha256(payload).hexdigest()!=DATA_SHA256:raise ValueError('Download hash mismatch')
        path.write_bytes(payload)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=DATA_SHA256:raise ValueError('Dataset hash mismatch')
    raw=sio.loadmat(path,spmatrix=True)
    conf_ids=[0,1,9,10,13];label_ids=[0,1,2,2,1]
    ids=np.flatnonzero(np.asarray(raw['PvsC'][:,conf_ids].sum(1)).ravel())
    conference=raw['PvsC'][ids].toarray().argmax(1)
    labels=np.array([dict(zip(conf_ids,label_ids))[c] for c in conference])
    features=raw['PvsT'][ids].toarray().astype('float32')
    features/=np.maximum(features.sum(1,keepdims=True),1)
    xs=[torch.from_numpy(features)];edges=[];node_ids=[ids.tolist()]
    for key in ['PvsA','PvsL']:
        incidence=raw[key][ids].tocsr();connected=np.flatnonzero(np.asarray(incidence.sum(0)).ravel())
        incidence=incidence[:,connected];a,b=incidence.nonzero()
        edge=torch.tensor(np.stack([a,b]),dtype=torch.long)
        edges.extend([edge,edge.flip(0)]);xs.append(torch.ones((len(connected),1)));node_ids.append(connected.tolist())
    rng=np.random.RandomState(99);parts=[[],[],[]]
    for c in range(3):
        perm=rng.permutation(np.flatnonzero(labels==c));n=len(perm);a=int(.2*n);b=int(.3*n)
        for acc,subset in zip(parts,[perm[:a],perm[a:b],perm[b:]]):acc.extend(subset.tolist())
    parts=[torch.tensor(sorted(v),dtype=torch.long) for v in parts]
    assert len(ids)==4025 and [len(t) for t in xs]==[4025,7167,60]
    assert len(torch.unique(torch.cat(parts)))==len(ids)
    assert [e.shape[1] for e in edges]==[13407,13407,4025,4025]
    meta={'data_url':DATA_URL,'data_sha256':DATA_SHA256,'nodes':[len(t) for t in xs],
          'features':[t.shape[1] for t in xs],'directed_edges':[e.shape[1] for e in edges],
          'raw_node_ids':node_ids,'split_seed':99,'split_sizes':[len(p) for p in parts],
          'split_sha256':hashlib.sha256(b''.join(p.numpy().tobytes() for p in parts)).hexdigest(),
          'graph_sha256':hashlib.sha256(b''.join(e.numpy().tobytes() for e in edges)).hexdigest(),
          'access':'transductive; all selected paper features/edges; labels only in masked objective/validation/scoring',
          'forbidden_input':'PvsC: conference incidence supplies labels only, never edges/features'}
    return {'x':xs,'edges':edges,'y':torch.tensor(labels),'train':parts[0],'val':parts[1],'test':parts[2],'meta':meta}

class RGCNLayer(nn.Module):
    """Eq2 relation-wise mean, sum over relations, one learned self transform, ReLU."""
    def __init__(self,width):
        super().__init__();self.weights=nn.Parameter(torch.empty(4,width,width));self.self_linear=nn.Linear(width,width)
        for w in self.weights:nn.init.xavier_uniform_(w)
    def forward(self,x,edges):
        out=[self.self_linear(t) for t in x]
        for r,(source,target) in enumerate(ROUTES):
            out[target]=out[target]+relation_mean(x[source]@self.weights[r],edges[r],len(x[target]))
        return [z.relu() for z in out]

class HGTLayer(nn.Module):
    """Typed multi-head attention with cross-relation normalization and gated residual.
    uniform=True removes Q/K, relation attention matrices and priors, retaining V,
    relation message matrices, A, skip and LayerNorm. No temporal encoding on ACM.
    """
    def __init__(self,width,heads=4,uniform=False):
        super().__init__();self.width=width;self.heads=heads;self.d=width//heads;self.uniform=uniform
        if width%heads:raise ValueError('width must be divisible by heads')
        if not uniform:
            self.k_linears=nn.ModuleList([nn.Linear(width,width) for _ in range(3)])
            self.q_linears=nn.ModuleList([nn.Linear(width,width) for _ in range(3)])
        self.v_linears=nn.ModuleList([nn.Linear(width,width) for _ in range(3)])
        self.a_linears=nn.ModuleList([nn.Linear(width,width) for _ in range(3)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        if not uniform:
            self.relation_pri=nn.Parameter(torch.ones(4,heads))
            self.relation_att=nn.Parameter(torch.empty(4,heads,self.d,self.d))
        self.relation_msg=nn.Parameter(torch.empty(4,heads,self.d,self.d));self.skip=nn.Parameter(torch.ones(3))
        if not uniform:nn.init.uniform_(self.relation_att,-math.sqrt(3/self.d),math.sqrt(3/self.d))
        nn.init.uniform_(self.relation_msg,-math.sqrt(3/self.d),math.sqrt(3/self.d))
    def forward(self,x,edges):
        values=[f(z).view(-1,self.heads,self.d) for f,z in zip(self.v_linears,x)]
        if not self.uniform:
            keys=[f(z).view(-1,self.heads,self.d) for f,z in zip(self.k_linears,x)]
            queries=[f(z).view(-1,self.heads,self.d) for f,z in zip(self.q_linears,x)]
        messages=[[],[],[]];scores=[[],[],[]];receivers=[[],[],[]]
        for r,(s,t) in enumerate(ROUTES):
            src,dst=edges[r]
            msg=torch.einsum('ehd,hdf->ehf',values[s][src],self.relation_msg[r])
            if self.uniform:score=msg.new_zeros((len(src),self.heads))
            else:
                key=torch.einsum('ehd,hdf->ehf',keys[s][src],self.relation_att[r])
                score=(queries[t][dst]*key).sum(-1)*self.relation_pri[r]/math.sqrt(self.d)
            messages[t].append(msg);scores[t].append(score);receivers[t].append(dst)
        out=[]
        for t in range(3):
            dst=torch.cat(receivers[t]);alpha=receiver_softmax(torch.cat(scores[t]),dst,len(x[t]))
            weighted=(torch.cat(messages[t])*alpha[:,:,None]).reshape(-1,self.width)
            aggregate=x[t].new_zeros(x[t].shape).index_add(0,dst,weighted)
            a=self.skip[t].sigmoid();out.append(self.norms[t](a*self.a_linears[t](F.gelu(aggregate))+(1-a)*x[t]))
        return out

class Model(nn.Module):
    """Same width32, depth2, adapters and classifier; family-specific hidden operators."""
    def __init__(self,inputs,arm,width=32):
        super().__init__();self.arm=arm
        if arm not in ARMS:raise ValueError(arm)
        self.adapters=nn.ModuleList([nn.Linear(d,width) for d in (inputs[:1] if arm=='mlp' else inputs)])
        if arm=='mlp':self.layers=nn.ModuleList([nn.Linear(width,width) for _ in range(2)])
        else:self.layers=nn.ModuleList([RGCNLayer(width) if arm=='rgcn' else HGTLayer(width,uniform=arm=='hgt_uniform') for _ in range(2)])
        self.classifier=nn.Linear(width,3)
    def forward(self,g):
        x=[torch.tanh(f(t)) for f,t in zip(self.adapters,g['x'])]
        for layer in self.layers:x=[layer(x[0]).relu()] if self.arm=='mlp' else layer(x,g['edges'])
        return self.classifier(x[0])

def classification_metrics(truth,prediction):
    """Accuracy and macro F1; three fixed classes, zero undefined F1."""
    truth=np.asarray(truth);prediction=np.asarray(prediction);f=[]
    for c in range(3):
        tp=np.sum((truth==c)&(prediction==c));fp=np.sum((truth!=c)&(prediction==c));fn=np.sum((truth==c)&(prediction!=c))
        f.append(float(2*tp/max(2*tp+fp+fn,1)))
    return {'accuracy':float(np.mean(truth==prediction)),'macro_f1':float(np.mean(f))}

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
import torch_geometric
import torch_geometric.typing as pyg_typing

TYPES = ('paper','author','subject')
EDGE_TYPES = (('paper','written_by','author'),('author','writes','paper'),
              ('paper','has_subject','subject'),('subject','describes','paper'))
BATCH_SIZE = 256
FANOUT = 8


def as_heterodata(g):
    """Preserve separate node-ID spaces; target labels never become features or edges."""
    data=HeteroData()
    for t,x in zip(TYPES,g['x']):data[t].x=x
    data['paper'].y=g['y']
    for e,edge in zip(EDGE_TYPES,g['edges']):data[e].edge_index=edge
    return data


def make_loader(data,seeds,fanout=FANOUT,batch_size=BATCH_SIZE):
    """Two directional hops for two layers; native sampling without replacement."""
    if not pyg_typing.WITH_PYG_LIB:raise RuntimeError('Native pyg-lib required; no stand-in sampler is permitted')
    return NeighborLoader(data,input_nodes=('paper',torch.as_tensor(seeds,dtype=torch.long)),
        num_neighbors={e:[fanout,fanout] for e in EDGE_TYPES},batch_size=batch_size,
        shuffle=False,replace=False,num_workers=0,subgraph_type='directional')


def local_graph(batch):
    """The same visible model accepts full or sampled local typed feature/edge arrays."""
    return {'x':[batch[t].x for t in TYPES],'edges':[batch[e].edge_index for e in EDGE_TYPES]}


def global_edges(batch,edge_type):
    """TODO 1: reconstruct global source and destination IDs using their own type maps."""
    source,_,target=edge_type
    a,b=batch[edge_type].edge_index
    return torch.stack([batch[source].n_id[a],batch[target].n_id[b]])


def seed_loss(logits,batch):
    """TODO 2: mean three-class cross-entropy over the first B seed papers only."""
    b=batch['paper'].batch_size
    return F.cross_entropy(logits[:b],batch['paper'].y[:b])


def batch_weight(batch_count,total_count):
    """TODO 3: weight a batch mean to recover the mean over all seeds, including a short final batch.
    Reject empty or oversized batches. This weights the audit and epoch report;
    the training optimizer still steps after each unweighted batch mean.
    """
    if not 0<batch_count<=total_count:raise ValueError('Require 0 < batch_count <= total_count')
    return batch_count/total_count


def initialize(g,arm,seed,width=32):
    """Copy every shared HGT tensor into the uniform ablation; seeds alone do not ensure this."""
    torch.manual_seed(seed);model=Model([x.shape[1] for x in g['x']],arm,width)
    if arm=='hgt_uniform':
        torch.manual_seed(seed);reference=Model([x.shape[1] for x in g['x']],'hgt',width)
        state=reference.state_dict();model.load_state_dict({k:state[k] for k in model.state_dict()})
    return model


def audit_batches(g,arm,batch_size,width=32,seeds=None):
    """All-neighbor seed logits and accumulated CE gradients must match a full-graph forward.
    Fixed weights, no optimizer steps, node-wise LayerNorm and no dropout.
    Float64 audit isolates algebra from float32 rounding. Tolerance fixed at 2e-6.
    """
    seeds=torch.arange(len(g['y'])) if seeds is None else torch.as_tensor(seeds)
    gd={**g,'x':[x.double() for x in g['x']]};data=as_heterodata(gd)
    model=initialize(gd,arm,100,width).double();reference=copy.deepcopy(model)
    full=reference(gd);F.cross_entropy(full[seeds],g['y'][seeds]).backward()
    gap=0.;seen=[];batch_counts=[]
    for batch in make_loader(data,seeds,-1,batch_size):
        b=batch['paper'].batch_size;ids=batch['paper'].n_id[:b];seen.extend(ids.tolist())
        logits=model(local_graph(batch));gap=max(gap,float((logits[:b]-full[ids]).abs().max().detach()))
        (seed_loss(logits,batch)*batch_weight(b,len(seeds))).backward()
        for e in EDGE_TYPES:assert torch.equal(global_edges(batch,e),data[e].edge_index[:,batch[e].e_id])
        batch_counts.append({'seeds':b,'nodes':sum(batch[t].num_nodes for t in TYPES),'edges':sum(batch[e].num_edges for e in EDGE_TYPES)})
    grad_gap=max(float(((a.grad if a.grad is not None else torch.zeros_like(a))-(b.grad if b.grad is not None else torch.zeros_like(b))).abs().max()) for a,b in zip(model.parameters(),reference.parameters()))
    assert seen==seeds.tolist() and len(set(seen))==len(seen)
    assert gap<2e-6 and grad_gap<2e-6,(arm,gap,grad_gap)
    return {'arm':arm,'batch_size':batch_size,'max_logit_gap':gap,'max_gradient_gap':grad_gap,'seed_coverage':seen,'batch_counts':batch_counts}


def fit(g,arm,lr,seed,epochs=EPOCHS,width=32,batch_size=BATCH_SIZE):
    """Identical epoch seed order/samples across arms and rates; validation only checkpointing.
    Training uses one Adam update per batch mean, NOT gradient accumulation.
    Evaluation uses the entire graph, which fits in memory for this dataset.
    """
    model=initialize(g,arm,seed,width);data=as_heterodata(g)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=.001)
    best=float('inf');checkpoint=None;trace=[];sample_hashes=[];start=time.perf_counter()
    max_nodes=0;max_edges=0
    for epoch in range(epochs):
        # Reset sampling RNG independently of initialization and model parameter count.
        torch.manual_seed(100000+seed*1000+epoch)
        order=g['train'][torch.randperm(len(g['train']))]
        model.train();total=0.;h=hashlib.sha256();seen=[];updates=0
        for batch in make_loader(data,order,FANOUT,batch_size):
            b=batch['paper'].batch_size;seen.extend(batch['paper'].n_id[:b].tolist())
            h.update(batch['paper'].n_id[:b].numpy().tobytes())
            for t in TYPES:h.update(batch[t].n_id.numpy().tobytes())
            for e in EDGE_TYPES:h.update(global_edges(batch,e).numpy().tobytes())
            optimizer.zero_grad();loss=seed_loss(model(local_graph(batch)),batch)
            loss.backward();optimizer.step();updates+=1
            total+=float(loss.detach())*batch_weight(b,len(order))
            max_nodes=max(max_nodes,sum(batch[t].num_nodes for t in TYPES))
            max_edges=max(max_edges,sum(batch[e].num_edges for e in EDGE_TYPES))
        assert seen==order.tolist() and len(set(seen))==len(order)
        sample_hashes.append(h.hexdigest());model.eval()
        with torch.no_grad():v=float(F.cross_entropy(model(g)[g['val']],g['y'][g['val']]))
        trace.append({'epoch':epoch+1,'train_ce':total,'val_ce':v,'updates':updates})
        if v<best:best=v;selected_epoch=epoch+1;checkpoint=copy.deepcopy(model.state_dict())
    model.load_state_dict(checkpoint);model.eval()
    return {'arm':arm,'lr':lr,'seed':seed,'epochs':epochs,'selected_epoch':selected_epoch,'best_val_ce':best,
        'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-start,
        'sample_hashes':sample_hashes,'trace':trace,'max_batch_nodes':max_nodes,'max_batch_edges':max_edges},model


def run_suite(root,output,progress=True):
    """Fresh complete 24-fit suite. Freeze choices before test scoring; no automatic resume."""
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    g=load_graph(root);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    checkpoint_dir=output.parent/(output.stem+'-checkpoints');checkpoint_dir.mkdir(exist_ok=True)
    result={'status':'RUNNING','protocol':{'arms':ARMS,'rates':LEARNING_RATES,'seeds':SEEDS,'epochs':EPOCHS,
        'width':32,'layers':2,'heads':4,'weight_decay':.001,'dropout':0,'split_seed':99,
        'batch_size':BATCH_SIZE,'fanout_per_relation_per_hop':FANOUT,'steps_per_epoch':4,
        'sampling_seed':'100000 + 1000*model_seed + zero_based_epoch',
        'selection':'mean best validation CE per arm; earliest checkpoint; tie smaller LR',
        'evaluation':'full graph; fixed full test set','scope':'new course experiment, not a published table'},
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'pyg':torch_geometric.__version__,
        'numpy':np.__version__,'platform':platform.platform(),'threads':2,'native_pyg_lib':pyg_typing.WITH_PYG_LIB},
        'data':g['meta'],'splits':{k:g[k].tolist() for k in ['train','val','test']},'runs':[],
        'paper_parity':'NOT_ESTABLISHED','learner_status':'PENDING_WRITTEN_DEFENSE'}
    def save():output.write_text(json.dumps(result,indent=2)+'\n')
    save()
    for arm in ARMS:
        for lr in LEARNING_RATES:
            for seed in SEEDS:
                rec,model=fit(g,arm,lr,seed)
                path=checkpoint_dir/f'{arm}-{lr}-{seed}.pt';torch.save(model.state_dict(),path)
                rec.update(checkpoint_file=path.name,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                result['runs'].append(rec);save()
                if progress:print(len(result['runs']),'/24',arm,lr,seed,'val',round(rec['best_val_ce'],4),'seconds',round(rec['seconds'],1),flush=True)
    for seed in SEEDS:
        hashes=[r['sample_hashes'] for r in result['runs'] if r['seed']==seed]
        assert all(h==hashes[0] for h in hashes),'Sample mismatch across comparison arms'
    result['selected_rates']={arm:choose_config(result['runs'],arm) for arm in ARMS};result['selected']=[]
    for rec in result['runs']:
        if rec['lr']!=result['selected_rates'][rec['arm']]:continue
        model=initialize(g,rec['arm'],rec['seed']);model.load_state_dict(torch.load(checkpoint_dir/rec['checkpoint_file'],weights_only=True));model.eval()
        with torch.no_grad():logits=model(g)[g['test']];pred=logits.argmax(1).tolist()
        row={k:rec[k] for k in ['arm','lr','seed','selected_epoch','parameters','seconds','max_batch_nodes','max_batch_edges']}
        row.update(classification_metrics(g['y'][g['test']].tolist(),pred));row.update(test_predictions=pred,test_logits=logits.tolist());result['selected'].append(row)
    result['test_labels']=g['y'][g['test']].tolist()
    result['summary']={arm:{key:{'mean':float(np.mean(v:=[r[key] for r in result['selected'] if r['arm']==arm])),
        'sample_sd':float(np.std(v,ddof=1))} for key in ['accuracy','macro_f1','seconds']} for arm in ARMS}
    result['paired_accuracy_differences']={a:[next(r['accuracy'] for r in result['selected'] if r['arm']=='hgt' and r['seed']==s)-next(r['accuracy'] for r in result['selected'] if r['arm']==a and r['seed']==s) for s in SEEDS] for a in ['rgcn','hgt_uniform','mlp']}
    result['paired_sampling']='EXACT';result['status']='COMPLETE';save();return result
