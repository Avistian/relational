"""Independent arithmetic and boundary checks for L071; run before training."""
import math
import numpy as np
import torch
torch.set_num_threads(1)
from relkit.vime_l071 import corrupt, pretext_loss, consistency_loss, make_split, VIME, pretrain, fit_predictor

def run():
    x = torch.tensor([[0.,1.],[1.,1.]])
    mask = torch.ones_like(x)
    donor = torch.tensor([[1,1],[0,0]])
    changed, xt = corrupt(x, mask, donor)
    assert torch.equal(xt, torch.tensor([[1.,1.],[0.,1.]]))
    assert torch.equal(changed, torch.tensor([[1.,0.],[1.,0.]])), 'Replacement can leave a value unchanged'
    assert torch.equal(corrupt(x,torch.zeros_like(x),donor)[1],x)
    z=torch.zeros_like(x,requires_grad=True)
    total,bce,mse=pretext_loss(z,torch.zeros_like(x),x,changed,2.)
    assert abs(bce.item()-math.log(2))<1e-6
    assert abs(mse.item()-.75)<1e-6, 'Reconstruct every coordinate'
    assert abs(total.item()-(math.log(2)+1.5))<1e-6
    total.backward(); assert z.grad.abs().sum()>0
    a=torch.tensor([[[0.,2.]],[[2.,4.]]],requires_grad=True)
    assert consistency_loss(a).item()==1., 'Population variance, across views only'
    assert consistency_loss(a[:1]).item()==0.
    assert consistency_loss(torch.ones(3,2,4)).item()==0.
    consistency_loss(a).backward(); assert a.grad.abs().sum()>0
    for seed in range(3):
        s=make_split(np.tile(np.arange(10),180),[50,150,500],seed)
        u=set(s['unlabeled']); te=set(s['test'])
        previous=set()
        for b in [50,150,500]:
            tr,va=s['budgets'][b]; selected=set(tr)|set(va)
            assert len(selected)==b and previous<=selected
            assert not set(tr)&set(va) and not selected&u and not selected&te and not u&te
            previous=selected
    torch.manual_seed(9)
    net=VIME(2); before={k:v.clone() for k,v in net.encoder.state_dict().items()}
    fitted,_=pretrain(x.repeat(20,1),9,epochs=2)
    assert any(not torch.equal(before[k],v) for k,v in fitted.encoder.state_dict().items())
    model,info=fit_predictor(fitted.encoder,x.repeat(20,1),torch.arange(40)%2,x,torch.tensor([0,1]),x,9,epochs=2,frozen=True,beta=1.)
    assert info['encoder_delta']==0., 'Frozen representation must stay fixed'
    print('PASS: corruption, collisions, full-coordinate loss, gradients, view-axis variance, nested disjoint splits, frozen encoder')

if __name__=='__main__':run()
