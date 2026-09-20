"""L099 visible controlled ACM comparison. Course experiment, not a paper table.
HGT follows the pinned modern pyHGT operator without RTE; all graph/task choices
and deviations are declared in l099-reproduction.md. No repository imports.
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
EPOCHS = 60


def relation_mean(x, edge, receivers):
    """TODO 1: mean incoming source rows per receiver; empty neighborhoods return zero."""
    src,dst=edge
    out=x.new_zeros((receivers,x.shape[1])).index_add(0,dst,x[src])
    degree=torch.bincount(dst,minlength=receivers).clamp_min(1)
    return out/degree[:,None]


def receiver_softmax(scores, dst, receivers):
    """TODO 2: stable E×heads softmax over ALL incoming relations per receiver/head."""
    ix=dst[:,None].expand_as(scores)
    maximum=scores.new_full((receivers,scores.shape[1]),-torch.inf)
    maximum.scatter_reduce_(0,ix,scores.detach(),reduce='amax',include_self=True)
    weights=(scores-maximum[dst]).exp()
    denominator=scores.new_zeros(maximum.shape).scatter_add(0,ix,weights)
    return weights/denominator[dst].clamp_min(torch.finfo(scores.dtype).tiny)


def choose_config(records, arm):
    """TODO 3: minimize mean best validation CE across seeds; ties use smaller LR."""
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


def fit(g,arm,lr,seed,epochs=EPOCHS,width=32):
    """All epochs run. Strict validation-CE improvement saves earliest minimum; no test use."""
    torch.manual_seed(seed);model=Model([x.shape[1] for x in g['x']],arm,width)
    # Pair common initial tensors for the attention ablation, despite removed scorer draws.
    if arm=='hgt_uniform':
        torch.manual_seed(seed);reference=Model([x.shape[1] for x in g['x']],'hgt',width)
        state=reference.state_dict();model.load_state_dict({k:state[k] for k in model.state_dict()})
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=.001)
    best=float('inf');checkpoint=None;trace=[];start=time.perf_counter()
    for epoch in range(epochs):
        model.train();optimizer.zero_grad();z=model(g);loss=F.cross_entropy(z[g['train']],g['y'][g['train']]);loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():v=float(F.cross_entropy(model(g)[g['val']],g['y'][g['val']]))
        trace.append({'epoch':epoch+1,'train_ce':float(loss.detach()),'val_ce':v})
        if v<best:best=v;selected_epoch=epoch+1;checkpoint=copy.deepcopy(model.state_dict())
    model.load_state_dict(checkpoint);model.eval()
    # Active means participates in the supervised graph; grad may numerically be zero.
    model.zero_grad();F.cross_entropy(model(g)[g['train']],g['y'][g['train']]).backward()
    active=sum(p.numel() for p in model.parameters() if p.grad is not None)
    record={'arm':arm,'lr':lr,'seed':seed,'epochs':epochs,'selected_epoch':selected_epoch,'best_val_ce':best,
            'parameters':sum(p.numel() for p in model.parameters()),'active_parameters':active,
            'seconds':time.perf_counter()-start,'trace':trace}
    return record,model


def classification_metrics(truth,prediction):
    """Accuracy and macro F1; three fixed classes, zero undefined F1."""
    truth=np.asarray(truth);prediction=np.asarray(prediction);f=[]
    for c in range(3):
        tp=np.sum((truth==c)&(prediction==c));fp=np.sum((truth!=c)&(prediction==c));fn=np.sum((truth==c)&(prediction!=c))
        f.append(float(2*tp/max(2*tp+fp+fn,1)))
    return {'accuracy':float(np.mean(truth==prediction)),'macro_f1':float(np.mean(f))}


def run_suite(root,output,progress=True):
    """Entire preregistered 24-fit search, then one frozen selection and test phase.
    Writes RUNNING checkpoints after each fit. No automatic resume or stale reuse.
    """
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    g=load_graph(root);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    checkpoint_dir=output.parent/(output.stem+'-checkpoints');checkpoint_dir.mkdir(exist_ok=True)
    result={'status':'RUNNING','protocol':{'arms':ARMS,'rates':LEARNING_RATES,'seeds':SEEDS,'epochs':EPOCHS,
        'width':32,'layers':2,'heads':4,'weight_decay':.001,'dropout':0,'split_seed':99,'selection':'mean best validation CE across seeds per arm; earliest minimum checkpoint','ablation_initialization':'HGT and uniform HGT share every common initial tensor'},
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'threads':2},
        'data':g['meta'],'splits':{k:g[k].tolist() for k in ['train','val','test']},'runs':[],
        'paper_parity':'NOT_ESTABLISHED','scope':'full declared course experiment, not a historical paper experiment'}
    def save():output.write_text(json.dumps(result,indent=2)+'\n')
    save()
    for arm in ARMS:
        for lr in LEARNING_RATES:
            for seed in SEEDS:
                rec,model=fit(g,arm,lr,seed)
                name=f'{arm}-{lr}-{seed}.pt';path=checkpoint_dir/name;torch.save(model.state_dict(),path)
                rec['checkpoint_file']=name;rec['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                result['runs'].append(rec);save()
                if progress:print(len(result['runs']),'/24',arm,lr,seed,'val',round(rec['best_val_ce'],4),'seconds',round(rec['seconds'],1),flush=True)
    result['selected_rates']={arm:choose_config(result['runs'],arm) for arm in ARMS}
    result['selected']=[]
    for rec in result['runs']:
        if rec['lr']!=result['selected_rates'][rec['arm']]:continue
        model=Model([x.shape[1] for x in g['x']],rec['arm'])
        model.load_state_dict(torch.load(checkpoint_dir/rec['checkpoint_file'],weights_only=True));model.eval()
        with torch.no_grad():pred=model(g)[g['test']].argmax(1).tolist()
        row={k:rec[k] for k in ['arm','lr','seed','selected_epoch','parameters','active_parameters','seconds']}
        row.update(classification_metrics(g['y'][g['test']].tolist(),pred));row['test_predictions']=pred;result['selected'].append(row)
    result['test_labels']=g['y'][g['test']].tolist()
    result['summary']={arm:{key:{'mean':float(np.mean(v:=[r[key] for r in result['selected'] if r['arm']==arm])),
                  'sample_sd':float(np.std(v,ddof=1))} for key in ['accuracy','macro_f1','seconds']} for arm in ARMS}
    result['paired_accuracy_differences']={other:[next(r['accuracy'] for r in result['selected'] if r['arm']=='hgt' and r['seed']==seed)-next(r['accuracy'] for r in result['selected'] if r['arm']==other and r['seed']==seed) for seed in SEEDS] for other in ['rgcn','hgt_uniform','mlp']}
    result['status']='COMPLETE';save();return result
