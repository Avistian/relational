"""Behavioral checks for the numeric RealMLP-TD-S teaching implementation."""
import numpy as np
import torch
from relkit.realmlp import RobustSmooth, coslog4, ntp_linear, last_best, RealMLPS

def check():
    x=np.array([[0,1,7],[0,2,7],[0,3,7],[0,4,7],[10,5,7]],dtype=float)
    prep=RobustSmooth().fit(x)
    out=prep.transform(np.array([[10,5,99]],dtype=float))
    np.testing.assert_allclose(out,[[2/np.sqrt(1+4/9),1/np.sqrt(1+1/9),0]])
    assert np.isfinite(out).all()
    for j in range(5): assert abs(coslog4((2**j-1)/15))<1e-12
    for j in range(4): assert abs(coslog4((2**(j+.5)-1)/15)-1)<1e-12
    assert last_best([.4,.2,.3,.2])==3
    a=torch.tensor([[1.,2.,3.,4.]],requires_grad=True)
    w=torch.ones(4,2,requires_grad=True); b=torch.tensor([[1.,-1.]],requires_grad=True)
    torch.testing.assert_close(ntp_linear(a,w,b),torch.tensor([[6.,4.]]))
    ntp_linear(a,w,b).sum().backward();assert a.grad.abs().sum()>0
    for regression in [False,True]:
        torch.manual_seed(53); m=RealMLPS(4,width=8,regression=regression)
        y=m(torch.randn(7,4));assert torch.count_nonzero(y)==0
        y.sum().backward()
        assert m.layers[-1].weight.grad.abs().sum()>0
        assert m.scale.grad.abs().sum()==0  # zero head blocks upstream gradients on first pass
    print('PASS: scaling edge cases, schedule extrema, tie selection, NTP gradients, zero-head behavior')

if __name__=='__main__':check()
