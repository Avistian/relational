"""B06 course implementation. Simplified generators and 2D learner, not Mitra parity."""
import hashlib
import numpy as np
import torch
from torch import nn

def choose_prior(u, p):
    """Outer mixture: one entire task comes from exactly one generator."""
    if not np.isfinite(u) or not np.isfinite(p) or not 0 <= u < 1 or not 0 <= p <= 1:
        raise ValueError('u must lie in [0,1); p in [0,1]')
    return 'scm' if u < p else 'tree'

def support_normalize(support, query):
    """Population statistics from support only; constant columns have scale1e-6."""
    support=np.asarray(support,dtype=np.float64);query=np.asarray(query,dtype=np.float64)
    mean=support.mean(axis=0);scale=np.maximum(support.std(axis=0),1e-6)
    return (support-mean)/scale,(query-mean)/scale

def paired_effect(left, right):
    """Match seed IDs before subtracting; positive means larger left metric."""
    if set(left)!=set(right) or not left:
        raise ValueError('Nonempty, identical paired identities required')
    return np.array([left[key]-right[key] for key in sorted(left)])

def make_task(family, seed, config):
    """Course proxy generators. Four features, noisy binary target; fixed support cut."""
    if family not in ['scm','tree','hybrid']:raise ValueError('Unknown family')
    rng=np.random.default_rng(seed);n=config['support']+config['query'];f=config['features']
    x=rng.normal(size=(n,f));weights=rng.normal(size=(f,f))
    # Lower-triangular structural equations create feature dependence.
    if family in ['scm','hybrid']:
        for j in range(1,f):x[:,j]+=np.tanh(x[:,:j]@weights[:j,j])/np.sqrt(j)
    neural=np.tanh(x@rng.normal(size=f))+0.3*np.sin(x[:,0]*rng.uniform(.5,2))
    axes=rng.integers(0,f,size=3);thresholds=rng.uniform(-.5,.5,size=3)
    root=x[:,axes[0]]>thresholds[0]
    child=np.where(root,x[:,axes[2]]>thresholds[2],x[:,axes[1]]>thresholds[1])
    leaves=rng.normal(size=4);tree=leaves[2*root.astype(int)+child.astype(int)]
    score={'scm':neural,'tree':tree,'hybrid':.5*neural+.5*tree}[family]
    score=score+.15*rng.normal(size=n)
    # Support-derived boundary avoids fitting any transform to query labels.
    y=(score>np.median(score[:config['support']])).astype(np.int64)
    a,b=support_normalize(x[:config['support']],x[config['support']:])
    return dict(x=np.concatenate([a,b]),y=y,raw_x=x)

class CellBlock(nn.Module):
    """Column self-attention then row attention with support-only keys/values."""
    def __init__(self,width,heads):
        super().__init__()
        self.column=nn.MultiheadAttention(width,heads,batch_first=True,dropout=0)
        self.row=nn.MultiheadAttention(width,heads,batch_first=True,dropout=0)
        self.n1=nn.LayerNorm(width);self.n2=nn.LayerNorm(width);self.n3=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))
    def forward(self,h,s):
        b,n,c,d=h.shape
        z=self.n1(h).reshape(b*n,c,d)
        h=h+self.column(z,z,z,need_weights=False)[0].reshape(b,n,c,d)
        z=self.n2(h).permute(0,2,1,3).reshape(b*c,n,d)
        update=self.row(z,z[:,:s],z[:,:s],need_weights=False)[0]
        h=h+update.reshape(b,c,n,d).permute(0,2,1,3)
        return h+self.ff(self.n3(h))

class CellLearner(nn.Module):
    """Small course 2D learner: scalar cells -> 2 blocks -> query target-token logits."""
    def __init__(self,config):
        super().__init__();d=config['width'];self.support=config['support']
        self.value=nn.Linear(1,d);self.label=nn.Embedding(3,d)
        self.blocks=nn.ModuleList([CellBlock(d,config['heads']) for _ in range(config['layers'])])
        self.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,2))
    def forward(self,x,support_y):
        b,n,f=x.shape;s=self.support
        if support_y.shape!=(b,s):raise ValueError('Pass support labels only')
        # Label token2 is a missing label, not a third output class.
        labels=torch.full((b,n),2,dtype=torch.long,device=x.device)
        labels[:,:s]=support_y
        h=torch.cat([self.value(x.unsqueeze(-1)),self.label(labels).unsqueeze(2)],dim=2)
        for block in self.blocks:h=block(h,s)
        return self.head(h[:,s:,-1])

def state_hash(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def train_one(config,arm,seed):
    """Matched fresh fit. Final checkpoint only; no test-driven selection."""
    torch.manual_seed(seed);model=CellLearner(config);initial=state_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    count=config['steps']*config['batch_tasks'];u=(np.arange(count)+.5)/count
    np.random.default_rng(seed+8000).shuffle(u)
    schedule=[choose_prior(float(v),config['arms'][arm]) for v in u]
    trace=[];model.train()
    for step in range(config['steps']):
        tasks=[]
        for j in range(config['batch_tasks']):
            i=step*config['batch_tasks']+j
            tasks.append(make_task(schedule[i],config['training_seed_base']+config['training_seed_stride']*seed+i,config))
        x=torch.tensor(np.stack([t['x'] for t in tasks]),dtype=torch.float32)
        y=torch.tensor(np.stack([t['y'] for t in tasks]))
        optimizer.zero_grad();logits=model(x,y[:,:config['support']])
        loss=nn.functional.cross_entropy(logits.flatten(0,1),y[:,config['support']:].flatten())
        loss.backward();optimizer.step();trace.append(float(loss.detach()))
    return model,dict(initial_sha256=initial,final_sha256=state_hash(model),loss=trace,prior_counts={f:schedule.count(f) for f in ['scm','tree']},schedule=schedule)

def predict_tasks(model,tasks,config):
    model.eval();s=config['support'];rows=[]
    with torch.no_grad():
        for task in tasks:
            x=torch.tensor(np.array(task['x'])[None],dtype=torch.float32)
            support_y=torch.tensor([task['y'][:s]],dtype=torch.long)
            logits=model(x,support_y)[0].double();prob=torch.softmax(logits,dim=-1).numpy()
            for j,(y,p) in enumerate(zip(task['y'][s:],prob)):
                rows.append(dict(task_id=task['id'],family=task['family'],query_id=s+j,y=int(y),p=p.tolist()))
    return rows
