"""Behavioral contracts for identity, ownership, ranking and full-run instrumentation."""
from pathlib import Path
import json
import torch
from relkit import identity_l132 as m

def rejected(fn):
    try:fn()
    except (ValueError,IndexError):return
    raise AssertionError('Invalid input was silently accepted')

def check_marker(fn):
    x=torch.zeros(5,2,requires_grad=True);marker=torch.tensor([[2.,-1.]],requires_grad=True)
    y=fn(x,2,marker)
    torch.testing.assert_close(y,torch.tensor([[2.,-1.],[2.,-1.],[0.,0.],[0.,0.],[0.,0.]]))
    y.sum().backward();torch.testing.assert_close(marker.grad,torch.tensor([[2.,2.]]))
    torch.testing.assert_close(x.detach(),torch.zeros(5,2))
    torch.testing.assert_close(x.grad,torch.ones(5,2))
    rejected(lambda:fn(x,6,marker));rejected(lambda:fn(x,2,torch.ones(1,3)))

def check_targets(fn):
    # Sponsor 7 is positive ONLY for owner 0; same global ID in owner 1 is negative.
    owner=torch.tensor([0,1,0,1]);ids=torch.tensor([7,7,8,9])
    y=fn(owner,ids,torch.tensor([0,1]),torch.tensor([7,9]),2)
    torch.testing.assert_close(y,torch.tensor([1.,0.,0.,1.]))
    rejected(lambda:fn(torch.tensor([2]),torch.tensor([7]),torch.tensor([0]),torch.tensor([7]),2))

def check_map(fn):
    assert abs(fn([[4,2,1],[2,3,0]],[[1,4],[3]],3)-2/3)<1e-12
    assert fn([[1,2]],[[5]],2)==0
    rejected(lambda:fn([[1,1]],[[1]],2))
    rejected(lambda:fn([[1,2]],[],2))

def main():
    check_marker(m.mark_roots);check_targets(m.candidate_targets);check_map(m.mean_average_precision)
    print('PASS: marker gradients, query ownership, MAP and malformed input')
if __name__=='__main__':main()
