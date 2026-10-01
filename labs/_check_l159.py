"""Behavioral contracts for the three live learner functions."""
import math
import torch

def check(mask_serializations, masked_cross_entropy, freeze_codec):
    tokens=torch.tensor([[2,4,6],[2,4,7]])
    identities=torch.tensor([[10,20,30],[10,20,31]])
    masked=mask_serializations(tokens,identities,20)
    assert masked.tolist()==[[2,0,6],[2,0,7]], 'Mask every copy of a schema target before encoding'
    assert tokens.tolist()==[[2,4,6],[2,4,7]], 'Do not modify clean targets in place'
    assert mask_serializations(tokens,identities,30).tolist()==[[2,4,0],[2,4,7]], 'Do not erase unrelated cells'
    for bad in [-1,99]:
        try: mask_serializations(tokens,identities,bad)
        except ValueError: pass
        else: raise AssertionError('Unknown target identity must be rejected')
    logits=torch.tensor([[[0.,math.log(3.)],[10.,-10.]]],requires_grad=True)
    loss=masked_cross_entropy(logits,torch.tensor([[1,1]]),torch.tensor([[True,False]]))
    assert abs(float(loss.detach())+math.log(.75))<1e-6, 'Loss must average selected targets only'
    loss.backward();assert torch.equal(logits.grad[0,1],torch.zeros(2)), 'Unselected logits must have zero gradient'
    try: masked_cross_entropy(logits,torch.tensor([[1,1]]),torch.zeros(1,2,dtype=torch.bool))
    except ValueError: pass
    else: raise AssertionError('An empty objective must fail, not yield NaN')
    codec=torch.nn.Linear(2,2)
    for p in codec.parameters():p.grad=torch.ones_like(p)
    freeze_codec(codec)
    assert all(not p.requires_grad and p.grad is None for p in codec.parameters())
    before={k:v.clone() for k,v in codec.state_dict().items()}
    h=torch.tensor([[.2,.3]],requires_grad=True)
    codec(h).square().sum().backward()
    assert h.grad is not None and h.grad.abs().sum()>0, 'Frozen weights must still transmit gradients'
    assert all(torch.equal(before[k],v) for k,v in codec.state_dict().items())
    return 'PASS: masking, selected loss, and frozen differentiable decoder'

if __name__=='__main__':
    from relkit.foundation_preview_l159 import mask_serializations,masked_cross_entropy,freeze_codec
    print(check(mask_serializations,masked_cross_entropy,freeze_codec))
