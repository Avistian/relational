"""Independent dense oracle and leakage/selection contracts, reusable in the lab."""
import json
from pathlib import Path
import torch
from relkit import ogb_l112 as base

def check_normalization(fn):
    # Duplicate and reverse edges are one relation, existing loops occur once;
    # node 3 is isolated and must preserve its own features.
    edge=torch.tensor([[0,0,1,1,2,2],[1,1,0,2,1,2]])
    got=fn(edge,4).to_dense()
    a=torch.tensor([[1.,1.,0.,0.],[1.,1.,1.,0.],[0.,1.,1.,0.],[0.,0.,0.,1.]])
    d=a.sum(1).sqrt();expected=a/d[:,None]/d[None,:]
    torch.testing.assert_close(got,expected)
    x=torch.arange(8,dtype=torch.float32).reshape(4,2).requires_grad_()
    (fn(edge,4)@x).square().sum().backward()
    torch.testing.assert_close(x.grad,2*expected.T@expected@x.detach())

def check_loss(fn):
    logits=torch.tensor([[1.,2.],[3.,-1.],[0.,1.]],requires_grad=True)
    y=torch.tensor([1,0,0]);train=torch.tensor([0])
    loss=fn(logits.log_softmax(1),y,train);loss.backward()
    assert torch.count_nonzero(logits.grad[1:])==0,'Held-out logits received loss gradients'
    changed=y.clone();changed[1:]=1-y[1:]
    torch.testing.assert_close(loss,fn(logits.log_softmax(1),changed,train))
    torch.testing.assert_close(loss,-logits.log_softmax(1)[0,1])

def check_selection(fn):
    assert fn([.6,.8,.7,.8])==1,'First maximum, not last/tied/test/final epoch'
    assert fn([.9,.8])==0
    try:fn([])
    except ValueError:pass
    else:raise AssertionError('Empty training history must fail')

if __name__=='__main__':
    for name,check in [('normalized_adjacency',check_normalization),('training_loss',check_loss),('selected_epoch',check_selection)]:
        check(getattr(base,name));print(name,'PASS')
    Path(__file__).with_name('_check_l112_results.json').write_text(json.dumps({'status':'PASS','checks':['dense_operator_and_gradient','heldout_label_invariance','first_validation_maximum']},indent=2))
