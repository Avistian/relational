"""Behavior contracts for composition, link direction, pooling and source-preserving refactor."""
import json
from pathlib import Path
import torch
from torch import nn
P=Path(__file__).resolve().parent

def check_compose(fn):
    calls=[]
    def encode(x):calls.append('encoder');return x+2
    def process(x,a):calls.append('processor');return a@x
    def head(x,a):calls.append('head');return a@x-1
    x=torch.tensor([[1.],[3.]]);a=torch.tensor([[0.,1.],[1.,0.]])
    torch.testing.assert_close(fn(encode,process,head,x,a),torch.tensor([[2.],[4.]]))
    assert calls==['encoder','processor','head']

def check_pairs(fn):
    x=torch.tensor([[1.,2.],[3.,5.],[7.,11.]],requires_grad=True)
    pairs=torch.tensor([[0,1,2],[1,0,2]])
    a=fn(x,pairs,False);b=fn(x,pairs,True)
    torch.testing.assert_close(a,torch.tensor([[3.,10.],[3.,10.],[49.,121.]]))
    torch.testing.assert_close(b,torch.tensor([[1.,2.,3.,5.],[3.,5.,1.,2.],[7.,11.,7.,11.]]))
    assert not torch.equal(b[0],b[1])
    b.sum().backward();torch.testing.assert_close(x.grad,torch.full_like(x,2.))

def check_pool(fn):
    x=torch.tensor([[1.,3.],[7.,9.],[3.,5.]],requires_grad=True);batch=torch.tensor([0,1,0])
    out=fn(x,batch,3)
    torch.testing.assert_close(out,torch.tensor([[2.,4.],[7.,9.],[0.,0.]]))
    perm=torch.tensor([2,0,1]);torch.testing.assert_close(fn(x[perm],batch[perm],3),out)
    out.sum().backward();torch.testing.assert_close(x.grad,torch.tensor([[.5,.5],[1.,1.],[.5,.5]]))
    torch.testing.assert_close(fn(x.detach().repeat_interleave(2,0),batch.repeat_interleave(2),3),out)

def check_modular():
    from relkit.patterns_l115 import GCN,normalized_adjacency,release_state
    from relkit.ogb_l112 import GCN as Flat
    torch.manual_seed(8);model=GCN(4,8,3);flat=Flat(4,8,3);flat.load_state_dict(release_state(model))
    edge=torch.tensor([[0,1,2,3],[1,2,0,4]]);adj=normalized_adjacency(edge,6);x=torch.randn(6,4)
    maxima={}
    for training in [False,True]:
        model.train(training);flat.train(training)
        a=x.clone().requires_grad_();b=x.clone().requires_grad_()
        torch.manual_seed(99);pa=model(a,adj)
        torch.manual_seed(99);pb=flat(b,adj)
        torch.testing.assert_close(pa,pb,rtol=0,atol=0)
        model.zero_grad();flat.zero_grad();pa.square().sum().backward();pb.square().sum().backward()
        torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
        for name,p in model.named_parameters():
            key=name.replace('processor.','').replace('head.conv.','convs.2.')
            torch.testing.assert_close(p.grad,dict(flat.named_parameters())[key].grad,rtol=0,atol=0)
        maxima[str(training)]=float((pa-pb).detach().abs().max())
    aopt=torch.optim.Adam(model.parameters(),lr=.01);bopt=torch.optim.Adam(flat.parameters(),lr=.01)
    aopt.step();bopt.step()
    for key,value in release_state(model).items():torch.testing.assert_close(value,flat.state_dict()[key],rtol=0,atol=0)
    assert sum(p.numel() for p in GCN().parameters())==110120
    return maxima

def check_boundaries():
    from relkit.patterns_l115 import GCN,normalized_adjacency,training_loss,selected_epoch
    torch.manual_seed(5);m=GCN(2,4,2,0.).eval()
    edges=torch.tensor([[0,1,2],[1,2,3]]);a=normalized_adjacency(edges,5);x=torch.randn(5,2)
    perm=torch.tensor([3,0,4,1,2]);inverse=torch.argsort(perm)
    perm_adj=normalized_adjacency(inverse[edges],5)
    with torch.no_grad():torch.testing.assert_close(m(x[perm],perm_adj),m(x,a)[perm],rtol=1e-5,atol=1e-6)
    logp=m(x,a);labels=torch.tensor([0,1,0,1,0]);idx=torch.tensor([0,1]);changed=labels.clone();changed[2:]=1-changed[2:]
    torch.testing.assert_close(training_loss(logp,labels,idx),training_loss(logp,changed,idx),rtol=0,atol=0)
    assert selected_epoch([.1,.5,.5,.2])==1

def run():
    from relkit import patterns_l115 as m
    assert hasattr(m,'graph_forward'),'Missing graph_forward behavior'
    check_compose(m.graph_forward);check_pairs(m.pair_features);check_pool(m.graph_mean);check_boundaries()
    return {'status':'PASS','composition':'PASS','directed_and_undirected_pairs':'PASS','pool_boundaries_and_gradients':'PASS','flat_refactor_output_errors':check_modular(),'flat_refactor_gradients_and_adam':'EXACT','permutation_and_label_visibility_and_first_tie':'PASS'}
if __name__=='__main__':
    r=run();(P/'_check_l115_results.json').write_text(json.dumps(r,indent=2));print(r)
