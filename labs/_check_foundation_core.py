"""Behavioral contracts for the 061–069 mechanism implementations.

These are counterexample/invariance checks, not a paper-training parity claim.
"""
import numpy as np
import torch
from torch.nn import functional as F
from relkit.foundation_core import (posterior_predictive, context_mask, attention,
    RowPFN, AxialPFN, induced_column, sample_scm, nearest_context, temporal_eligible,
    crossfit_embeddings, open_class_loss)


def checks():
    torch.set_num_threads(1)
    assert posterior_predictive([1, 0, 1], 2, 3) == 0.5
    mask = context_mask(3, 2)
    assert mask.shape == (5, 5) and mask[:, :3].all() and not mask[:, 3:].any()
    torch.manual_seed(61)
    q, k, v = torch.randn(2, 3, 4), torch.randn(2, 5, 4), torch.randn(2, 5, 6)
    ours = attention(q, k, v)
    torch.testing.assert_close(ours, F.scaled_dot_product_attention(q, k, v))
    for cls in (RowPFN, AxialPFN):
        model = cls(3, width=16, heads=2, layers=2).eval()
        x, y, query = torch.randn(1, 5, 3), torch.tensor([[0,1,0,1,1]]), torch.randn(1,2,3)
        base = model(x, y, query)
        perm = torch.tensor([4,1,0,3,2])
        torch.testing.assert_close(base, model(x[:,perm], y[:,perm], query), atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(base[:,:1], model(x,y,query[:,:1]), atol=2e-6, rtol=2e-5)
        more = torch.cat([query, torch.full((1,1,3), 1000.)], 1)
        torch.testing.assert_close(base, model(x,y,more)[:,:2], atol=2e-6, rtol=2e-5)
        base.sum().backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    u,ind = torch.randn(7,8),torch.randn(3,8)
    a=induced_column(u,ind,4)
    changed=u.clone();changed[5:]=999
    b=induced_column(changed,ind,4)
    torch.testing.assert_close(a[:5],b[:5])
    noise=np.random.default_rng(63).normal(size=(20,3));w=np.array([[0.,2.,0.],[0.,0.,-1.],[0.,0.,0.]])
    z=sample_scm(noise,w,lambda x:x)
    np.testing.assert_allclose(z[:,1],2*z[:,0]+noise[:,1])
    np.testing.assert_allclose(z[:,2],-z[:,1]+noise[:,2])
    ids=nearest_context(np.array([[0.],[1.],[2.]]),np.array([[1.]]),2,np.array([1]))
    assert ids.tolist()==[[0,2]],'Exclude row identity even when its distance is zero'
    assert temporal_eligible([1,2,3],[2,8,4],4).tolist()==[True,False,True]
    seen=[]
    def embed(cx,cy,qx):
        seen.append((set(cx[:,0]),set(qx[:,0])))
        return qx*2
    x=np.arange(12.).reshape(6,2)
    result=crossfit_embeddings(x,np.arange(6)%2,np.array([0,1,0,1,0,1]),embed)
    np.testing.assert_allclose(result,x*2)
    assert all(not a&b for a,b in seen)
    assert np.isfinite(open_class_loss([0,2],[[.8,.2],[.3,.7]],[0,1]))
    print('PASS: posterior, SDPA parity, context isolation, permutation, gradients, SCM, retrieval, time, cross-fitting, open classes')


if __name__ == '__main__': checks()
