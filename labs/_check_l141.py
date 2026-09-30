"""Independent learner contracts; reject plausible wrong implementations."""
import math
import numpy as np
import torch

def check_routes(fn):
    edges=[('race','f2p_circuit','circuit'),('result','f2p_driver','driver'),('result','f2p_constructor','constructor'),('result','f2p_race','race')]
    out=fn(edges)
    assert len(out)==8, 'One ordinary FK gives two directions; three FKs give six ordered composite routes'
    assert ('dim-dim','race','f2p_circuit','circuit') in out
    assert ('dim-dim','circuit','rev_f2p_circuit','race') in out
    assert ('dim-fact-dim','result','f2p_driver','driver','constructor','rev_f2p_constructor','result') in out
    assert len(set(out))==8
    assert fn(edges+[(d,'rev_'+r,s) for s,r,d in edges])==out,'Reverse edges must not become foreign keys'
    assert len(fn([('result','f2p_home','team'),('result','f2p_away','team')]))==2,'Foreign key roles, not distinct target tables, identify routes'
    assert fn([])==[]

def check_softmax(fn):
    s=torch.tensor([[0.,1.],[math.log(3),1.],[1000.,-1000.]],dtype=torch.float64,requires_grad=True)
    dest=torch.tensor([0,0,2]);a=fn(s,dest,4)
    expected=torch.tensor([[.25,.5],[.75,.5],[1.,1.]],dtype=s.dtype)
    torch.testing.assert_close(a,expected,atol=1e-12,rtol=1e-12)
    p=torch.tensor([2,0,1]);torch.testing.assert_close(fn(s[p],dest[p],4),expected[p])
    assert fn(s[:0],dest[:0],4).shape==(0,2)
    (a*torch.arange(6).reshape(3,2)).sum().backward();assert torch.isfinite(s.grad).all()

def check_mae(fn):
    q=[(1,10),(1,20),(2,10)]
    assert fn(q,[1.,4.,2.],q[::-1],[3.,2.,1.])==1.
    for keys,pred in [(q[:2],[1.,2.]),([q[0],q[0],q[2]],[1.,2.,3.]),(q,[1.,float('nan'),3.])]:
        try:fn(q,[1.,4.,2.],keys,pred)
        except ValueError:pass
        else:raise AssertionError('Missing/duplicate/nonfinite prediction must fail')

if __name__=='__main__':
    from relkit.relgnn_l141 import get_atomic_routes,destination_softmax,keyed_mae
    check_routes(get_atomic_routes);check_softmax(destination_softmax);check_mae(keyed_mae)
    print('PASS route semantics, destination/head normalization, keyed MAE')
