"""Independent arithmetic, graph-boundary and checkpoint-verdict tests."""
import json
from pathlib import Path
import numpy as np
import torch
from relkit.checkpoint_l090 import eligible_neighbors, draw_neighbors, masked_mean, verdict, SampledSAGE
from relkit.gcn_l082 import normalized_support, propagate
import scipy.sparse as sp

def verify():
    torch.set_num_threads(1)
    a=sp.csr_matrix([[0,1,0,0],[1,0,1,0],[0,1,0,0],[0,0,0,0]])
    lists=eligible_neighbors(a,np.array([0,1,3]))
    assert lists[1].tolist()==[0] and len(lists[2])==0
    ids,mask=draw_neighbors(np.array([1,3]),lists,4,np.random.default_rng(1))
    assert set(ids[mask].tolist())=={0}
    values=torch.tensor([[[2.],[8.]],[[99.],[99.]]],requires_grad=True)
    m=masked_mean(values,torch.tensor([[True,True],[False,False]]))
    torch.testing.assert_close(m,torch.tensor([[5.],[0.]]));m.sum().backward()
    torch.testing.assert_close(values.grad,torch.tensor([[[.5],[.5]],[[0.],[0.]]]))
    s=normalized_support(a);h=torch.tensor([[2.],[4.],[8.],[9.]])
    expected=torch.tensor([[1+4/np.sqrt(6)],[10/np.sqrt(6)+4/3],[4+4/np.sqrt(6)],[9.]],dtype=torch.float32)
    torch.testing.assert_close(propagate(s,h,torch.ones(1,1)),expected)
    assert verdict(.815,100,True)=='CLOSE'
    assert verdict(.90,100,True)=='FAIL'
    assert verdict(.815,3,True)=='INCOMPLETE'
    assert verdict(.815,100,False)=='INCOMPARABLE'
    # Shared weights: deterministic degree-one fixture makes sampled and full inference identical.
    pair=sp.csr_matrix([[0,1,0],[1,0,0],[0,0,0]])
    lists=eligible_neighbors(pair,np.arange(3));x=torch.randn(3,4);model=SampledSAGE(4,3)
    roots=np.arange(3)
    z=model.sampled(x,roots,lists,np.random.default_rng(4),fanout=2)
    torch.testing.assert_close(z,model.full(x,lists))
    # Even extreme unavailable features cannot affect a training computation.
    lists=eligible_neighbors(a,np.array([0,1,3]));x=torch.randn(4,4)
    z=model.sampled(x,np.array([1]),lists,np.random.default_rng(8))
    x[2]=1e6
    torch.testing.assert_close(z,model.sampled(x,np.array([1]),lists,np.random.default_rng(8)))
    return {'status':'PASS','checks':['GCN dense hand oracle including isolate','eligible node exclusion','empty-neighbor zero and gradient','degree-one sampled/full logits','held-out feature intervention','protocol before tolerance verdict']}
if __name__=='__main__':
    r=verify();Path(__file__).with_name('_verify_l090_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
