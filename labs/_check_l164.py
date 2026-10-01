"""Learner contracts: fixed counterexamples precede implementation."""
import torch

def check_attention(fn):
    q=torch.tensor([[[1.,0.]]],dtype=torch.float64)
    k=torch.tensor([[[1.,0.],[0.,1.],[20.,20.]]],dtype=torch.float64)
    v=torch.tensor([[[2.,0.],[0.,4.],[999.,999.]]],dtype=torch.float64)
    blocked=torch.tensor([[[False,False,True]]])
    want=torch.tensor([[[1.339523098653314,1.320953802693373]]],dtype=torch.float64)
    torch.testing.assert_close(fn(q,k,v,blocked),want,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(fn(q,k[:,[1,0,2]],v[:,[1,0,2]],blocked),want,atol=1e-12,rtol=1e-12)
    try:fn(q,k,v,torch.ones_like(blocked))
    except ValueError:pass
    else:raise AssertionError('All-masked attention must be rejected')

def check_relations(fn):
    x=torch.tensor([[0.,0.],[2.,4.],[4.,2.],[8.,1.]])
    ei=torch.tensor([[0,0,0],[1,2,3]]);r=torch.tensor([0,0,1]);e=torch.tensor([[1.,1.],[.5,2.]])
    want=torch.tensor([[4.,3.],[0.,0.],[0.,0.],[0.,0.]])
    torch.testing.assert_close(fn(x,ei,r,e),want)
    # Repeat EVERY neighbour of one relation: its mean and final result stay fixed.
    ei2=torch.tensor([[0,0,0,0,0],[1,2,1,2,3]]);r2=torch.tensor([0,0,0,0,1])
    torch.testing.assert_close(fn(x,ei2,r2,e),want)
    torch.testing.assert_close(fn(x,torch.empty(2,0,dtype=torch.long),torch.empty(0,dtype=torch.long),e),torch.zeros_like(x))
    # Max over present negative messages must remain negative, not clamp to zero.
    torch.testing.assert_close(fn(-x,ei,r,e)[0],torch.tensor([-3.,-2.]))

def check_cutoffs(fn):
    times=torch.tensor([[4,5,6,-1],[4,5,6,-1]])
    cutoff=torch.tensor([5,7])
    assert torch.equal(fn(times,cutoff),torch.tensor([[True,False,False,False],[True,True,True,False]]))
    try:fn(times,torch.tensor([5]))
    except ValueError:pass
    else:raise AssertionError('A batch-wide cutoff must not broadcast silently')

def check_all(attention,pool,cutoffs):
    check_attention(attention);check_relations(pool);check_cutoffs(cutoffs)
    return 'PASS: masked attention, relation balance, per-owner time'

if __name__=='__main__':
    from relkit.griffin_l164 import cell_attention,relation_pool,eligible_edges
    print(check_all(cell_attention,relation_pool,eligible_edges))
