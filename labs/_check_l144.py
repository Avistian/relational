"""Behavioral contracts: owner isolation, gradient routing, temporal cutoffs, MAP."""
import numpy as np
import torch
from relkit.context_l144 import fuse_scores,audit_cutoffs,keyed_map

def check_fusion(fn):
    tower=torch.tensor([[1.,2.,3.],[4.,5.,6.]],requires_grad=True)
    local=torch.tensor([10.,20.],requires_grad=True)
    offset=torch.tensor([.5,-.5],requires_grad=True)
    y=fn(tower,local,torch.tensor([0,1]),torch.tensor([1,1]),offset)
    assert torch.equal(y,torch.tensor([[1.,10.5,3.],[4.,19.5,6.]]))
    assert torch.equal(tower.detach(),torch.tensor([[1.,2.,3.],[4.,5.,6.]]))
    y.sum().backward()
    assert torch.equal(tower.grad,torch.tensor([[1.,0.,1.],[1.,0.,1.]]))
    assert torch.equal(local.grad,torch.ones(2)) and torch.equal(offset.grad,torch.ones(2))
    assert torch.equal(fn(tower,local[:0],torch.tensor([],dtype=torch.long),torch.tensor([],dtype=torch.long),offset),tower)

def check_cutoffs(fn):
    assert fn(np.array([5,10]),np.array([0,1]),np.array([5,10]))==2
    try:fn(np.array([6,10]),np.array([0,1]),np.array([5,10]))
    except (AssertionError,ValueError):pass
    else:raise AssertionError('Batch maximum leaked a future event')

def check_map(fn):
    keys=[(3,10),(3,20)];truth=[[2,4],[5]]
    # first query: hit at rank1 and3 => (1 + 2/3)/2; second =>1/2
    got=fn(keys,truth,keys[::-1],np.array([[9,5,8],[2,9,4]]),3)
    assert abs(got-2/3)<1e-12
    for pk,p in [(keys[:1],np.array([[2,9,4]])),([keys[0],keys[0]],np.array([[2,9,4],[9,5,8]])),(keys,np.array([[2,2,4],[9,5,8]]))]:
        try:fn(keys,truth,pk,p,3)
        except (ValueError,AssertionError):pass
        else:raise AssertionError('Invalid ranking/key set accepted')

if __name__=='__main__':
    check_fusion(fuse_scores);check_cutoffs(audit_cutoffs);check_map(keyed_map);print('PASS: fusion, temporal ownership, keyed MAP')
