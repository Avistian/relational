"""Behavioral contracts; call these on learner functions, not copied solutions."""
import numpy as np
import torch

def check_mix(fn):
    parts=[torch.full((2,3,2),float(i)) for i in range(5)]
    layer=torch.nn.Linear(10,2,bias=False)
    with torch.no_grad():layer.weight.copy_(torch.arange(20).reshape(2,10))
    expected=torch.tensor([130.,330.]).expand(2,3,2)
    assert torch.equal(fn(parts,layer),expected)
    x=[p.clone().requires_grad_() for p in parts];fn(x,layer).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in x)
    try:fn(parts[:4],layer)
    except ValueError:pass
    else:raise AssertionError('Missing token element accepted')

def check_audit(fn):
    keys=[(7,5),(7,10)]
    assert fn(keys,[[4,5],[6,10]])==[]
    assert fn(keys,[[4,6],[6,10]])==[(0,1)]
    try:fn([(7,5),(7,5)],[[4],[4]])
    except ValueError:pass
    else:raise AssertionError('Duplicate query keys accepted')

def check_selection(fn):
    assert fn([4.,3.,3.,5.])==2
    assert fn([2.,3.])==0
    for bad in [[],[1.,float('nan')]]:
        try:fn(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid score history accepted')

if __name__=='__main__':
    from relkit.relgt_contracts_l145 import mix_five,audit_tokens,last_validation_min
    check_mix(mix_five);check_audit(audit_tokens);check_selection(last_validation_min)
    print('PASS: three learner contracts')
