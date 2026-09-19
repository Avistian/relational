"""Independent fixtures for the load-bearing L072 operations."""
import math
import torch
from relkit.contrastive_l072 import corrupt, scarf_loss, subtab_loss, aggregate_views, subsets, draw_view, SubTab, load_split, probe

def run():
    x=torch.tensor([[1.,10.,100.],[2.,20.,200.]])
    bank=torch.tensor([[3.,30.,300.],[4.,40.,400.]])
    mask=torch.tensor([[True,False,True],[False,True,False]])
    donors=torch.tensor([[1,0,0],[0,1,1]])
    assert torch.equal(corrupt(x,bank,mask,donors),torch.tensor([[4.,10.,300.],[2.,40.,200.]]))
    assert torch.equal(x,torch.tensor([[1.,10.,100.],[2.,20.,200.]])), 'Do not mutate clean input'
    z=torch.eye(2,requires_grad=True)
    loss=scarf_loss(z,z,1.)
    assert abs(loss.item()-math.log1p(math.exp(-1)))<1e-6, 'Positive remains in N-way denominator'
    assert abs(scarf_loss(torch.ones(3,2),torch.ones(3,2)).item()-math.log(3))<1e-6
    loss.backward();assert z.grad.abs().sum()>0
    # SubTab sees two off-diagonal negatives, not one; self-similarity excluded.
    assert abs(subtab_loss(torch.eye(2),torch.eye(2),1.).item()-math.log1p(2/math.e))<1e-6
    h=torch.tensor([[[1.,3.],[2.,4.]],[[5.,7.],[6.,8.]]])
    assert torch.equal(aggregate_views(h),torch.tensor([[3.,5.],[4.,6.]])), 'Average views, preserve rows'
    assert subsets(12)==[[0,1,2,3,4,5,6],[1,2,3,4,5,6,7],[5,6,7,8,9,10,11]]
    torch.manual_seed(5)
    a=torch.ones(8,5);b=torch.zeros(12,5)
    assert torch.all((draw_view(a,b,.6)!=a).sum(1)==3), 'Select exactly floor(c*d) features'
    assert torch.equal(draw_view(a,b,0.),a)
    model=SubTab(12).eval();rows=torch.randn(4,12)
    assert torch.allclose(model.represent(rows)[:1],model.represent(rows[:1]),atol=1e-6), 'Rows must not influence each other at inference'
    x,y,split=load_split('wine',0)
    original=probe(x,y,split);changed=y.copy();changed[split['test']]=(changed[split['test']]+1)%3
    intervention=probe(x,changed,split)
    assert original['C']==intervention['C'] and original['trials']==intervention['trials']
    assert original['prediction']==intervention['prediction'], 'Test labels must not affect fitted predictions'
    print('PASS: donor-column identity, no mutation, exact losses, gradients, aggregation, subset layout')
if __name__=='__main__':run()
