"""Independent arithmetic contracts for live learner functions."""
import torch,numpy as np

def check_edge_sum(fn):
    values=torch.tensor([[2.,3.],[5.,7.],[11.,13.]],requires_grad=True)
    out=fn(values,torch.tensor([1,0,1]),3)
    torch.testing.assert_close(out,torch.tensor([[5.,7.],[13.,16.],[0.,0.]]))
    out.square().sum().backward()
    torch.testing.assert_close(values.grad,torch.tensor([[26.,32.],[10.,14.],[26.,32.]]))
    assert fn(values[:0],torch.empty(0,dtype=torch.long),3).shape==(3,2)

def check_fuse(fn):
    source=torch.tensor([[2.],[7.]],requires_grad=True);fact=torch.tensor([[10.],[20.],[30.]],requires_grad=True)
    edges=torch.tensor([[1,0,1],[0,1,2]])
    out=fn(source,fact,edges)
    torch.testing.assert_close(out,torch.tensor([[17.],[22.],[37.]]))
    out.sum().backward();torch.testing.assert_close(source.grad,torch.tensor([[1.],[2.]]))
    torch.testing.assert_close(fact.grad,torch.ones_like(fact))
    torch.testing.assert_close(fn(source,fact,edges[:,:0]),fact)

def check_pair(fn):
    q=[(1,10),(1,20),(2,10)];y=[1.,5.,8.]
    actual=fn(q,y,q,[2.,3.,8.],q[::-1],[10.,4.,1.])
    np.testing.assert_allclose(actual,[ -1.,-1.,2.]) # ordinary error minus composite error
    for args in [(q,y,q,[2.,3.,8.],q[:-1],[1.,2.]),(q,y,q,[2.,3.,8.],[q[0]]*3,[1.,2.,3.]),(q,y,q,[float('nan'),3.,8.],q,[1.,2.,3.])]:
        try:fn(*args)
        except ValueError:pass
        else:raise AssertionError('Invalid paired population accepted')

if __name__=='__main__':
    from relkit.pathology_l142 import edge_sum,route_fuse,paired_loss_gap
    check_edge_sum(edge_sum);check_fuse(route_fuse);check_pair(paired_loss_gap);print('PASS three live contracts')
