"""Independent small-graph oracles; also used by live notebook CHECKs."""
import json
from pathlib import Path
import torch

def check_mean(fn):
    edge=torch.tensor([[0,2,1,2],[1,1,2,0]])
    a=fn(edge,4)
    expected=torch.tensor([[0.,0.,1.,0.],[.5,0.,.5,0.],[0.,1.,0.,0.],[0.,0.,0.,0.]])
    torch.testing.assert_close(a.to_dense(),expected)
    x=torch.arange(8.).reshape(4,2).requires_grad_()
    z=torch.sparse.mm(a,x);z.sum().backward()
    torch.testing.assert_close(x.grad,expected.T@torch.ones(4,2))
    # Duplicate edges and raw self-edges count; do not silently binarize them.
    repeated=torch.tensor([[0,0,2,1],[1,1,1,1]])
    actual=fn(repeated,3).to_dense()
    oracle=torch.tensor([[0.,0.,0.],[.5,.25,.25],[0.,0.,0.]])
    torch.testing.assert_close(actual,oracle)

def check_induced(fn):
    edge=torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]])
    # Reverse node order deliberately catches missing global-to-local remapping.
    got=fn(edge,torch.tensor([2,1]),4)
    assert set(map(tuple,got.T.tolist()))=={(0,1),(1,0)}
    assert fn(edge,torch.tensor([0,3]),4).shape==(2,0)

def check_selection(fn):
    assert fn([{'epoch':20,'valid':.8,'test':.99},{'epoch':25,'valid':.9,'test':.1},{'epoch':30,'valid':.9,'test':.8}])==1
    try:fn([])
    except ValueError:pass
    else:raise AssertionError('Empty history must fail')

def check_model(module):
    torch.manual_seed(113)
    model=module.SAGE(3,5,2,3,.5).eval()
    edge=torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]])
    a=module.mean_adjacency(edge,5);x=torch.randn(5,3,requires_grad=True)
    actual=model(x,a)
    z=x
    for i,c in enumerate(model.convs):
        z=(a.to_dense()@z)@c.lin_l.weight.T+c.lin_l.bias+z@c.lin_r.weight.T
        if i<2:z=z.relu()
    torch.testing.assert_close(actual,z.log_softmax(-1))
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(),x)[0],torch.autograd.grad(z.log_softmax(-1).square().sum(),x)[0])
    for batch in [1,2,5]:
        logits=module.layerwise_inference(model,x.detach(),a,batch,'cpu')
        torch.testing.assert_close(logits.log_softmax(-1),actual)
    # Held-out labels cannot affect a fixed training loss.
    y=torch.tensor([0,1,0,1,0]);mask=torch.tensor([True,True,False,False,False]);other=y.clone();other[~mask]=1-other[~mask]
    torch.testing.assert_close(module.masked_loss(actual,y,mask),module.masked_loss(actual,other,mask))

def main():
    import importlib.util
    target=Path(__file__).parent/'relkit/scaling_l113.py'
    assert target.exists(),'Missing visible L113 implementation'
    from relkit import scaling_l113 as m
    check_mean(m.mean_adjacency);check_induced(m.induced_edges);check_selection(m.selected_epoch);check_model(m)
    result={'status':'PASS','checks':['directed mean and isolated node','input gradients','induced relabeling and boundary cut','first validation maximum','dense model and gradient oracle','exact layerwise inference three chunk sizes','held-out label mutation']}
    (Path(__file__).parent/'_check_l113_results.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
