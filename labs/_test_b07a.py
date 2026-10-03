"""Independent hand-computed oracles, then source parity and invalid-input tests."""
import math
import torch
from relkit.hyper_b07a import class_weights,retrieval_bias,break_even

def check_class_weights(fn):
    p=torch.tensor([[1.,2.,.1],[3.,4.,.3],[5.,6.,.5]])
    h=torch.tensor([[2.,0.],[0.,2.],[1.,1.]])
    y=torch.tensor([0,0,1]);before=p.clone()
    expected=torch.tensor([[3.,6.,4.],[4.,7.,5.],[.2,.5,.3]])
    torch.testing.assert_close(fn(p,h,y,3),expected)
    torch.testing.assert_close(p,before)
    order=torch.tensor([2,0,1]);torch.testing.assert_close(fn(p[order],h[order],y[order],3),expected)

def check_retrieval(fn):
    logits=torch.tensor([[1.,2.],[4.,1.],[0.,0.]])
    query=torch.tensor([[0.,0.],[2.,2.],[1.,0.]])
    support=torch.tensor([[0.,0.],[2.,0.],[2.,2.]])
    labels=torch.tensor([0,1,1]);before=logits.clone()
    out=fn(logits,query,support,labels,torch.tensor(.75))
    torch.testing.assert_close(out,torch.tensor([[1.75,2.],[4.,1.75],[.75,0.]]))
    torch.testing.assert_close(logits,before)

def check_break_even(fn):
    assert fn(2,.001,0,.005)==501,'Strictly cheaper requires501 queries, not500'
    assert fn(2,.001,0,.005,refreshes=2)==1501
    assert fn(0,.001,2,.001)==0
    assert fn(2,.005,0,.001) is None
    assert fn(0,.005,2,.001)==0
    for args in [(-1,0,0,0),(1,float('nan'),0,1),(1,1,1,1,-1)]:
        try:fn(*args)
        except ValueError:pass
        else:raise AssertionError('Invalid costs accepted')

if __name__=='__main__':
    check_class_weights(class_weights);check_retrieval(retrieval_bias);check_break_even(break_even)
    print('Learner oracles PASS')
