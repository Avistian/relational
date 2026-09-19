"""Independent behavioral oracles for L085; run before experiments."""
import json
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"relkit"))
from oversmoothing_l085 import support, smooth, collapse_metrics, DeepGCN

def check():
    a=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=float)
    s,d=support(a)
    expected=torch.tensor([[.5,6**-.5,0],[6**-.5,1/3,6**-.5],[0,6**-.5,.5]],dtype=torch.float64)
    torch.testing.assert_close(s,expected)
    h=torch.tensor([[2.],[4.],[8.]],dtype=torch.float64)
    torch.testing.assert_close(smooth(s,h,1),expected@h)
    q=d.sqrt();q=q/q.norm()
    limit=q[:,None]*(q@h)[None,:]
    torch.testing.assert_close(smooth(s,h,200),limit)
    assert float(limit.var())>.01, 'Symmetric limit need not have identical rows'
    assert collapse_metrics(smooth(s,h,200),d)['degree_variance']<1e-20
    disconnected=np.zeros((4,4));disconnected[0,1]=disconnected[1,0]=1;disconnected[2,3]=disconnected[3,2]=1
    ss,dd=support(disconnected)
    distinct=torch.tensor([[0.],[0.],[2.],[2.]],dtype=torch.float64)
    torch.testing.assert_close(smooth(ss,distinct,200),distinct)
    assert collapse_metrics(distinct,dd)['degree_variance']>.1
    # Scaling all coordinates cannot manufacture an angular collapse.
    z=torch.tensor([[1.,0.],[0.,1.],[1.,1.]],dtype=torch.float64)
    assert abs(collapse_metrics(z,d)['mean_cosine']-collapse_metrics(z*1e-4,d)['mean_cosine'])<1e-12
    assert collapse_metrics(torch.zeros_like(z),d)['mean_cosine'] is None
    model=DeepGCN(3,2,3).double().eval()
    x=torch.eye(3,dtype=torch.float64);manual=x
    for i,w in enumerate(model.weights):
        manual=expected@manual@w
        if i<len(model.weights)-1:manual=manual.relu()
    torch.testing.assert_close(model(x,s),manual)
    p=torch.tensor([2,0,1]);torch.testing.assert_close(model(x[p],s[p][:,p]),manual[p])
    return {'status':'PASS','checks':['exact normalized weights','200-step spectral limit','degree-scaled nonconstant limit','disconnected component counterexample','scale invariant angular metric','zero-vector undefined cosine','complete forward oracle','node permutation']}

if __name__=='__main__':
    result=check();Path(__file__).with_name('_verify_l085_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
