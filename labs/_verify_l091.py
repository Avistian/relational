"""Independent arithmetic, gradient, graph-access and optimizer checks."""
import importlib.util,json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('rgcn_l091',P/'relkit/rgcn_l091.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(1)
a=sp.csr_matrix(([1.,1.],([0,0],[1,2])),shape=(3,3))
b=sp.csr_matrix(([1.],([0],[2])),shape=(3,3))
s=m.normalize_relations([a,b,sp.eye(3)])
x=torch.tensor([[1.],[2.],[8.]])
w=torch.tensor([[[2.]], [[-1.]], [[1.]]],requires_grad=True)
y=m.relation_sum(s,x,w)
assert float(y[0].detach())==3.,y
assert float(y[1].detach())==2.
y.sum().backward();torch.testing.assert_close(w.grad.flatten(),torch.tensor([5.,8.,11.]))
v=torch.randn(2,3,4,requires_grad=True);c=torch.randn(3,2,requires_grad=True)
z=m.compose_weights(v,c);oracle=torch.stack([sum(c[r,k]*v[k] for k in range(2)) for r in range(3)])
torch.testing.assert_close(z,oracle)
z.square().sum().backward();assert v.grad.abs().sum()>0 and c.grad.abs().sum()>0
# Permuting relation order and weights together is invariant; weights alone are not.
torch.testing.assert_close(m.relation_sum(s,x,w),m.relation_sum([s[1],s[0],s[2]],x,w[[1,0,2]]))
assert not torch.allclose(m.relation_sum(s,x,w),m.relation_sum(s,x,w[[1,0,2]]))
# Featureless optimization is exactly the explicit identity operation on a tiny graph.
net=m.RGCN(3,3,2,hidden=4,bases=2)
logits=net(s)
ws1=m.compose_weights(net.v1,net.c1);ws2=m.compose_weights(net.v2,net.c2)
expected=m.relation_sum(s,torch.relu(m.relation_sum(s,torch.eye(3),ws1)),ws2)
torch.testing.assert_close(logits,expected)
params=list(net.parameters());g1=torch.autograd.grad(logits.square().sum(),params,retain_graph=True);g2=torch.autograd.grad(expected.square().sum(),params)
for g,h in zip(g1,g2):torch.testing.assert_close(g,h)
y=torch.tensor([0,1,1]);idx=torch.tensor([0]);l=m.masked_loss(logits,y,idx);y[1:]=1-y[1:];torch.testing.assert_close(l,m.masked_loss(logits,y,idx))
# Keras Adam: epsilon inside uncorrected second-moment denominator.
p=torch.nn.Parameter(torch.tensor([1.]));opt=m.KerasAdam([p]);p.grad=torch.tensor([2.]);opt.step()
expected=1-.01*(np.sqrt(.001)/.1)*.2/(np.sqrt(.004)+1e-8)
assert abs(p.item()-expected)<1e-7
result={'status':'PASS','checks':['per-relation mean arithmetic','message gradient oracle','basis reconstruction and gradients','relation permutation and intervention','implicit identity forward and gradient oracle','held-out label loss isolation','historical Adam first step']}
(P/'_verify_l091_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
