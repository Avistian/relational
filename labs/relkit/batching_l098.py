"""Complete L098 synthetic experiment; native PyG sampling, visible model/trainer."""
import copy
import hashlib
import json
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
import torch_geometric.typing as pyg_typing
TYPES=('customer','orders','product')
EDGES=(('orders','buyer','customer'),('customer','rev_buyer','orders'),('orders','item','product'),('product','rev_item','orders'))
CONFIG={'graphs':list(range(32)),'batch_sizes':[1,4,24],'fit_seeds':[0,1,2],'fanouts':[-1,2,1],'epochs':40,'width':16,'lr':0.02,'cutoff':5}

def make_graph(seed=0):
    """24 customers, 144 order events (days 1..6), 12 static products; no target features."""
    gen=torch.Generator().manual_seed(seed)
    g=HeteroData()
    for t,n in zip(TYPES,[24,144,12]):
        g[t].x=torch.randn(n,4,generator=gen)
        g[t].time=torch.zeros(n,dtype=torch.long)
    g['orders'].time=torch.arange(1,7).repeat(24)
    customer=torch.arange(24).repeat_interleave(6)
    product=torch.randint(12,(144,),generator=gen)
    order=torch.arange(144)
    for e,ix in [(EDGES[0],torch.stack([order,customer])),(EDGES[2],torch.stack([order,product]))]:
        g[e].edge_index=ix;g[EDGES[EDGES.index(e)+1]].edge_index=ix.flip(0)
    # Synthetic target: sign of average product feature 0 in days <=5.
    # This is deliberately learnable from the graph, not a future-outcome benchmark.
    score=g['product'].x[product,0].reshape(24,6)[:,:5].mean(1)
    g['customer'].y=(score>0).float()
    return g

def snapshot(g,cutoff):
    """Keep stable global row IDs; remove all edges touching unavailable event nodes."""
    out=g.clone()
    for src,rel,dst in g.edge_types:
        e=(src,rel,dst);a,b=g[e].edge_index
        keep=(g[src].time[a]<=cutoff)&(g[dst].time[b]<=cutoff)
        out[e].edge_index=g[e].edge_index[:,keep]
    return out

def make_loader(g,seeds,fanout=-1,batch_size=4,times=None):
    """Two hops for two layers; relation-specific budgets; no replacement or hidden fallback."""
    if not pyg_typing.WITH_PYG_LIB:
        raise RuntimeError('Install a torch-compatible pyg-lib build; this lab tests real native sampling.')
    options={} if times is None else dict(input_time=torch.as_tensor(times),time_attr='time',disjoint=True,temporal_strategy='uniform')
    return NeighborLoader(g,input_nodes=('customer',torch.as_tensor(seeds,dtype=torch.long)),
        num_neighbors={e:[fanout,fanout] for e in g.edge_types},batch_size=batch_size,
        shuffle=False,replace=False,num_workers=0,subgraph_type='directional',**options)

def relation_mean(x,edge_index,n_dst):
    """TODO 1: mean incoming source features for each destination, with zero for empty rows."""
    src,dst=edge_index
    total=x.new_zeros((n_dst,x.shape[1]));count=x.new_zeros(n_dst)
    total.index_add_(0,dst,x[src]);count.index_add_(0,dst,x.new_ones(dst.numel()))
    return total/count.clamp_min(1).unsqueeze(1)

class TypedMean(nn.Module):
    """Two shared-across-batches typed layers; one root transform plus one per relation."""
    def __init__(self,width=16):
        super().__init__()
        self.roots=nn.ModuleList([nn.ModuleDict({t:nn.Linear(d,width) for t in TYPES}) for d in [4,width]])
        self.messages=nn.ModuleList([nn.ModuleDict({str(i):nn.Linear(d,width,bias=False) for i in range(4)}) for d in [4,width]])
        self.head=nn.Linear(width,1)
    def forward(self,g):
        h=g.x_dict
        for roots,messages in zip(self.roots,self.messages):
            z={t:roots[t](h[t]) for t in TYPES}
            for i,e in enumerate(EDGES):
                src,_,dst=e
                z[dst]=z[dst]+messages[str(i)](relation_mean(h[src],g[e].edge_index,h[dst].shape[0]))
            h={t:F.relu(z[t]) for t in TYPES}
        return self.head(h['customer']).flatten()

def seed_loss(logits,batch):
    """TODO 2: BCE only for the first B seed customers; context customers are not targets."""
    b=batch['customer'].batch_size
    return F.binary_cross_entropy_with_logits(logits[:b],batch['customer'].y[:b])

def global_edges(batch,edge_type):
    """TODO 3: restore per-type global endpoints; never treat local edge IDs as table row IDs."""
    src,_,dst=edge_type;a,b=batch[edge_type].edge_index
    return torch.stack([batch[src].n_id[a],batch[dst].n_id[b]])

def dense_oracle(model,g):
    """Independent dense adjacency forward: no index_add or relation_mean."""
    h=g.x_dict
    for roots,messages in zip(model.roots,model.messages):
        z={t:roots[t](h[t]) for t in TYPES}
        for i,(src,rel,dst) in enumerate(EDGES):
            a,b=g[src,rel,dst].edge_index
            mat=h[src].new_zeros((h[dst].shape[0],h[src].shape[0]))
            for u,v in zip(a.tolist(),b.tolist()):mat[v,u]+=1
            mat=mat/mat.sum(1,keepdim=True).clamp_min(1)
            z[dst]=z[dst]+messages[str(i)](mat@h[src])
        h={t:F.relu(z[t]) for t in TYPES}
    return model.head(h['customer']).flatten()

def audit_graph(seed,batch_size):
    """Compare all seed logits and weighted accumulated gradients with a dense full graph."""
    torch.set_num_threads(1);torch.manual_seed(seed)
    g=snapshot(make_graph(seed),5);model=TypedMean();oracle=copy.deepcopy(model)
    expected=dense_oracle(oracle,g)
    F.binary_cross_entropy_with_logits(expected,g['customer'].y).backward()
    logit_gap=0.;seen=[];batch_nodes=[];batch_edges=[]
    for batch in make_loader(g,range(24),-1,batch_size):
        b=batch['customer'].batch_size;ids=batch['customer'].n_id[:b];seen+=ids.tolist()
        logits=model(batch);logit_gap=max(logit_gap,float((logits[:b]-expected[ids]).abs().max().detach()))
        (seed_loss(logits,batch)*b/24).backward()
        batch_nodes.append(sum(batch[t].num_nodes for t in TYPES))
        batch_edges.append(sum(batch[e].num_edges for e in EDGES))
        for e in EDGES:
            restored=global_edges(batch,e)
            # e_id indexes the original unsorted edge store, despite internal CSC sorting.
            assert torch.equal(restored,g[e].edge_index[:,batch[e].e_id])
            assert set(map(tuple,restored.t().tolist()))<=set(map(tuple,g[e].edge_index.t().tolist()))
    assert all((a.grad is None)==(b.grad is None) for a,b in zip(model.parameters(),oracle.parameters()))
    gap=max(float((a.grad-b.grad).abs().max()) for a,b in zip(model.parameters(),oracle.parameters()) if a.grad is not None)
    assert sorted(seen)==list(range(24)) and len(set(seen))==24
    return dict(graph_seed=seed,batch_size=batch_size,max_logit_gap=logit_gap,max_gradient_gap=gap,seed_coverage=seen,max_batch_nodes=max(batch_nodes),max_batch_edges=max(batch_edges))

def temporal_audit():
    """Two queries for customer 0 at days 2 and 5; compare to separate filtered snapshots."""
    torch.manual_seed(98);g=make_graph(3);model=TypedMean()
    batch=next(iter(make_loader(g,[0,0],-1,2,[2,5])))
    assert batch['customer'].n_id[:2].tolist()==[0,0]
    assert batch['customer'].batch[:2].tolist()==[0,1]
    expected=torch.stack([dense_oracle(model,snapshot(g,t))[0] for t in [2,5]])
    gap=float((model(batch)[:2]-expected).abs().max().detach());assert gap<2e-6
    for t in TYPES:
        assert (g[t].time[batch[t].n_id]<=torch.tensor([2,5])[batch[t].batch]).all()
    for e in EDGES:
        a,b=batch[e].edge_index
        assert torch.equal(batch[e[0]].batch[a],batch[e[2]].batch[b])
    # Future changes cannot affect an earlier seed. This includes static product hazards:
    # only future event features are changed here; static products must themselves be available.
    changed=g.clone();changed['orders'].x[changed['orders'].time>2]+=10000
    early=next(iter(make_loader(changed,[0],-1,1,[2])))
    assert torch.allclose(model(early)[0],expected[0],atol=2e-6)
    return {'status':'PASS','query_ids':[0,0],'cutoffs':[2,5],'max_logit_gap':gap,'event_ids':{str(k):batch['orders'].n_id[batch['orders'].batch==k].tolist() for k in [0,1]}}

def train(seed,fanout):
    """Fixed graph/splits/budget; validation chooses checkpoint; test evaluated once afterward."""
    torch.manual_seed(seed);g=snapshot(make_graph(98),5);model=TypedMean()
    opt=torch.optim.Adam(model.parameters(),lr=CONFIG['lr']);best=float('inf');state=None;history=[]
    for epoch in range(CONFIG['epochs']):
        model.train();total=0.
        for batch in make_loader(g,range(16),fanout,4):
            opt.zero_grad();loss=seed_loss(model(batch),batch);loss.backward();opt.step();total+=float(loss.detach())/4
        model.eval()
        with torch.no_grad():v=F.binary_cross_entropy_with_logits(model(g)[16:20],g['customer'].y[16:20]).item()
        history.append({'epoch':epoch+1,'train_bce':total,'val_bce':v})
        if v<best:best=v;state=copy.deepcopy(model.state_dict());chosen=epoch+1
    model.load_state_dict(state);model.eval()
    with torch.no_grad():
        all_logits=model(g);test=all_logits[20:];labels=g['customer'].y[20:]
        loss=F.binary_cross_entropy_with_logits(test,labels).item();acc=((test>=0)==labels.bool()).float().mean().item()
        gaps=[];nodes=[];edges=[]
        for batch in make_loader(g,range(24),fanout,4):
            b=batch['customer'].batch_size;ids=batch['customer'].n_id[:b]
            gaps.extend((model(batch)[:b]-all_logits[ids]).abs().tolist())
            nodes.append(sum(batch[t].num_nodes for t in TYPES));edges.append(sum(batch[e].num_edges for e in EDGES))
    return {'seed':seed,'fanout':fanout,'selected_epoch':chosen,'val_bce':best,'test_bce':loss,'test_accuracy':acc,'mean_abs_sampled_logit_gap':sum(gaps)/len(gaps),'mean_batch_nodes':sum(nodes)/len(nodes),'mean_batch_edges':sum(edges)/len(edges),'test_logits':test.tolist(),'test_labels':labels.tolist(),'history':history,'state_dict':{k:v.tolist() for k,v in model.state_dict().items()}}

def restore_model(run):
    """Restore the selected trained model from transparent JSON arrays for fresh inference."""
    model=TypedMean(CONFIG['width'])
    model.load_state_dict({k:torch.tensor(v,dtype=model.state_dict()[k].dtype) for k,v in run['state_dict'].items()})
    return model.eval()


def run_suite():
    """All 96 correctness configurations, native temporal test, nine complete training fits."""
    torch.set_num_threads(1)
    records=[audit_graph(s,b) for s in CONFIG['graphs'] for b in CONFIG['batch_sizes']]
    assert max(r['max_logit_gap'] for r in records)<2e-6
    assert max(r['max_gradient_gap'] for r in records)<2e-6
    temporal=temporal_audit();runs=[train(s,f) for f in CONFIG['fanouts'] for s in CONFIG['fit_seeds']]
    for run in runs:
        with torch.no_grad(): replay=restore_model(run)(snapshot(make_graph(98),5))[20:]
        assert torch.allclose(replay,torch.tensor(run['test_logits']),atol=1e-7)
    return {'status':'PASS','config':CONFIG,'audit':records,'temporal':temporal,'runs':runs,'scope':'Complete synthetic course experiment; no RelBench benchmark or paper score reproduction','paper_parity':'NOT_ESTABLISHED','relbench_benchmark':'NOT_RUN'}
