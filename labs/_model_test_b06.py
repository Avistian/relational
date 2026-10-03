"""End-to-end leakage, equivariance and trainable signal tests."""
import json
from pathlib import Path
import numpy as np
import torch
from relkit.prior_b06 import make_task,CellLearner

def check_model(config):
    torch.set_num_threads(1);torch.manual_seed(0)
    for family in ['scm','tree','hybrid']:
        t=make_task(family,31,config)
        assert t['x'].shape==(32,4) and t['y'].shape==(32,)
        assert set(t['y'][:24])=={0,1}
        assert np.array_equal(t['x'],make_task(family,31,config)['x'])
    model=CellLearner(config).eval()
    t=make_task('scm',7,config)
    x=torch.tensor(t['x'][None],dtype=torch.float32);y=torch.tensor(t['y'][None,:24])
    logits=model(x,y);assert logits.shape==(1,8,2)
    altered=x.clone();altered[:,25:]*=500
    torch.testing.assert_close(logits[:,0],model(altered,y)[:,0],atol=1e-6,rtol=1e-5)
    # Support reordering and feature reordering preserve the prediction function.
    order=torch.arange(23,-1,-1)
    perm=torch.cat([order,torch.arange(24,32)])
    torch.testing.assert_close(logits,model(x[:,perm],y[:,order]),atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(logits,model(x[:,:,[3,1,0,2]],y),atol=2e-6,rtol=1e-5)
    assert not torch.allclose(logits,model(x,1-y)), 'Support labels ignored'
    loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),torch.tensor(t['y'][24:]))
    loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print('PASS: generators, query isolation, permutation invariance, label influence, gradients')

if __name__=='__main__':check_model(json.loads((Path(__file__).parent/'evidence/b06/course-protocol.json').read_text()))
