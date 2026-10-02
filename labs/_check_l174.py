"""Behavioral contracts; accepts live student policies, not copied answers."""
import numpy as np
import torch
from torch import nn

def check174(policy, split_fn, select_fn, model_cls):
    tasks=[dict(classes=1,kind='number'),dict(classes=3,kind='category')]
    for arm in ['freeze','full','adapter','scratch']:
        torch.manual_seed(5);model=model_cls(tasks,arm)
        policy(model,arm)
        before={k:v.clone() for k,v in model.state_dict().items()}
        opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.01)
        context=torch.tensor([[1,2,0,0,0,0,0,0],[2,1,0,0,0,0,0,0]])
        columns=torch.tensor([0,1,2]);numeric=torch.tensor([0.,2.,-1.]);categories=torch.tensor([0,0,1])
        for _ in range(3):
            out=model(context,torch.tensor([0,1]),columns,numeric,categories)
            loss=(out[:,0]-3).square().sum();opt.zero_grad();loss.backward();opt.step()
        changed={k for k,v in model.state_dict().items() if not torch.equal(v,before[k])}
        assert any(k.startswith('heads.') for k in changed)
        if arm in ['freeze','adapter']:
            assert not any(not k.startswith(('heads.','adapter.')) for k in changed),'Frozen backbone changed'
            assert all(not p.requires_grad for k,p in model.named_parameters() if not k.startswith(('heads.','adapter.')))
        else:assert any(k.startswith('shared.') for k in changed),'Full training did not update encoder'
        if arm=='adapter':assert any(k.startswith('adapter.') for k in changed),'Adapter cannot learn'
    torch.manual_seed(2);plain=model_cls(tasks,'freeze')
    torch.manual_seed(2);adapt=model_cls(tasks,'adapter')
    x=torch.randn(7,32);assert torch.equal(adapt.adapter(x),x),'Adapter must start at identity'
    assert torch.equal(plain(context,torch.tensor([0,1]),columns,numeric,categories),adapt(context,torch.tensor([0,1]),columns,numeric,categories))
    dates=np.array(['2005-12-31','2006-01-01','2006-12-31','2007-01-01','2007-12-31','2008-01-01'],dtype='datetime64[s]').astype('int64')
    assert np.array_equal(split_fn(dates),[-1,0,0,1,1,2])
    assert select_fn([.4,.2,.2])==1
    for invalid in [[],[np.nan],[np.inf]]:
        try:select_fn(invalid)
        except ValueError:pass
        else:raise AssertionError('Invalid validation scores accepted')
    try:policy(plain,'typo')
    except ValueError:pass
    else:raise AssertionError('Unknown arm accepted')
    return 'PASS'

if __name__=='__main__':
    from pathlib import Path
    assert (Path(__file__).parent/'relkit/finetune_l174.py').exists(),'L174 adaptation implementation missing'
    from relkit.finetune_l174 import configure_trainable,adaptation_split,select_epoch,AdaptationModel
    print(check174(configure_trainable,adaptation_split,select_epoch,AdaptationModel))
