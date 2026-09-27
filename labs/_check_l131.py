"""Behavioral contracts include temporal alignment, root cardinality and graph preservation."""
import json
from pathlib import Path
import torch
from relkit.stack_l131 import activation_summary,relative_days,seed_readout

def rejected(fn):
    try:fn()
    except (ValueError,TypeError):return
    raise AssertionError('Expected rejection of invalid input')

def check_summary(fn):
    x=torch.tensor([[1.,-2.],[0.,3.]],requires_grad=True)
    s=fn(x)
    assert s['shape']==[2,2] and s['mean']==.5 and s['max_abs']==3.
    assert s['zero_fraction']==.25 and s['first_rows']==[[1.,-2.],[0.,3.]]
    assert x.requires_grad and x.grad is None
    assert fn(torch.empty(0,2))['mean'] is None
    rejected(lambda:fn(torch.tensor([[float('nan')]])))

def check_days(fn):
    seed=torch.tensor([864000,1728000]);times=torch.tensor([777600,1555200,691200]);owner=torch.tensor([0,1,0])
    torch.testing.assert_close(fn(seed,times,owner),torch.tensor([1.,2.,2.]),rtol=0,atol=0)
    rejected(lambda:fn(seed,torch.tensor([1800000]),torch.tensor([1])))
    rejected(lambda:fn(seed,times,torch.tensor([0,2,0])))
    rejected(lambda:fn(seed,times,torch.tensor([0,1])))

def check_readout(fn):
    x=torch.arange(15,dtype=torch.float32).reshape(5,3).requires_grad_()
    y=fn(x,2);assert y.shape==(2,3)
    torch.testing.assert_close(y,x[:2]);y.sum().backward()
    torch.testing.assert_close(x.grad,torch.tensor([[1.,1.,1.],[1.,1.,1.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]))
    rejected(lambda:fn(x,6));rejected(lambda:fn(x,0));rejected(lambda:fn(x,1.5))

if __name__=='__main__':
    for fn,check in [(activation_summary,check_summary),(relative_days,check_days),(seed_readout,check_readout)]:check(fn)
    r=dict(status='PASS',contracts=['activation arithmetic and finite values','per-query clock alignment and future rejection','root count and intact gradient graph'])
    Path(__file__).with_name('_check_l131_results.json').write_text(json.dumps(r,indent=2));print(r)
