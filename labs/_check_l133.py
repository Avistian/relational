"""Behavioral contracts: arithmetic, empty/absent, independent source and gradients."""
import copy,json
from pathlib import Path
import torch
from torch_geometric.nn import HeteroConv,SAGEConv
from relkit.hetero_l133 import sum_neighbors,relation_output,merge_relations

def check_neighbors(fn):
    x=torch.tensor([[1.,2.],[3.,4.],[9.,8.]],requires_grad=True)
    e=torch.tensor([[0,1,1],[0,0,2]])
    y=fn(x,e,4)
    torch.testing.assert_close(y,torch.tensor([[4.,6.],[0.,0.],[3.,4.],[0.,0.]]))
    torch.testing.assert_close(fn(x,torch.tensor([[1,1],[0,0]]),1),torch.tensor([[6.,8.]]))
    y.sum().backward();torch.testing.assert_close(x.grad,torch.tensor([[1.,1.],[2.,2.],[0.,0.]]))
    torch.testing.assert_close(fn(x,torch.empty((2,0),dtype=torch.long),4),torch.zeros(4,2))
    try:fn(x,torch.tensor([[0],[-1]]),4)
    except ValueError:pass
    else:raise AssertionError('negative destination accepted')

def check_relation(fn):
    c=SAGEConv((1,1),1,aggr='sum')
    with torch.no_grad():c.lin_l.weight.fill_(2);c.lin_l.bias.fill_(1);c.lin_r.weight.fill_(3)
    x=torch.tensor([[1.],[2.]]);d=torch.tensor([[4.],[5.]])
    e=torch.tensor([[0,1],[0,0]])
    torch.testing.assert_close(fn(c,x,d,e),torch.tensor([[19.],[16.]]))
    torch.testing.assert_close(fn(c,x,d,torch.empty((2,0),dtype=torch.long)),torch.tensor([[13.],[16.]]))

def check_merge(fn):
    a=torch.tensor([[1.,2.]],requires_grad=True);b=torch.tensor([[3.,7.]],requires_grad=True)
    o=fn({('a','r','d'):a,('b','s','d'):b,('d','rev','a'):b})
    torch.testing.assert_close(o['d'],torch.tensor([[4.,9.]]));assert set(o)=={'d','a'}
    o['d'].sum().backward();torch.testing.assert_close(a.grad,torch.ones_like(a));torch.testing.assert_close(b.grad,torch.ones_like(b))
    assert fn({})=={}

if __name__=='__main__':
    check_neighbors(sum_neighbors);check_relation(relation_output);check_merge(merge_relations)
    print('PASS: three live arithmetic contracts')
