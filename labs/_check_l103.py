"""Behavioral oracles for the live lesson mechanisms."""
import json, math
from pathlib import Path
import numpy as np
import torch
from relkit.tgat_l103 import encode_time, attention_weights, strict_prefix, NeighborFinder, TGAT

torch.set_num_threads(1)
x=torch.tensor([[0.,2.]],requires_grad=True);w=torch.tensor([1.,.5],requires_grad=True);p=torch.zeros(2,requires_grad=True)
y=encode_time(x,w,p)
assert y.shape==(1,2,2) and torch.allclose(y[0,0],torch.ones(2))
y.sum().backward();assert w.grad.abs().sum()>0 and p.grad.abs().sum()>0
q=torch.tensor([[[1.,0.]]]);k=torch.tensor([[[1.,0.],[0.,1.],[9.,9.]]]);mask=torch.tensor([[[False,False,True]]])
a=attention_weights(q,k,mask)
assert a[0,0,2]==0 and torch.allclose(a.sum(-1),torch.ones(1,1))
assert abs(a[0,0,0].item()-1/(1+math.exp(-1/math.sqrt(2))))<1e-6
assert torch.allclose(attention_weights(q,k,torch.ones_like(mask)),torch.full((1,1,3),1/3))
for ts in [[],[1],[1,1,2,4],[0,3,9]]:
 for cutoff in [-1,0,1,2,3,4,10]:
  assert strict_prefix(np.array(ts),cutoff)==sum(t<cutoff for t in ts)
print('PASS: time encoding, gradients, scaled masking, release padding, strict boundaries')
# Independent adjacency oracle: no production search helper in expected values.
rng=np.random.RandomState(14);cases=0
for graph_id in range(30):
 u=rng.randint(1,6,20);v=rng.randint(6,10,20);t=np.sort(rng.randint(0,15,20)).astype(float);ev={'u':u,'v':v,'t':t,'e':np.arange(1,21)}
 finder=NeighborFinder(ev,10,release=False,uniform=False)
 for node in range(1,10):
  for cutoff in [0.,3.,7.,15.]:
   expected=[(int(v[i] if u[i]==node else u[i]),i+1,float(t[i])) for i in range(20) if (u[i]==node or v[i]==node) and t[i]<cutoff]
   assert finder.find_before(node,cutoff).tolist()==[list(x) for x in expected]
   cases+=1
# Instrument the actual encoder to prove the three tasks affect a differentiable forward pass.
import relkit.tgat_l103 as model_module
calls={name:0 for name in ['encode_time','attention_weights','strict_prefix']}
originals={name:getattr(model_module,name) for name in calls}
for name,fun in originals.items():
 def counted(*args,_name=name,_fun=fun,**kw):calls[_name]+=1;return _fun(*args,**kw)
 setattr(model_module,name,counted)
ev={'u':np.array([1,1,2,1]),'v':np.array([3,4,3,4]),'t':np.array([1.,3.,4.,6.]),'e':np.arange(1,5)}
finder=NeighborFinder(ev,5,release=False,uniform=False)
model=TGAT(finder,np.zeros((5,4),np.float32),rng.normal(size=(5,4)).astype('float32'),layers=2,dropout=0)
p,n=model.contrast(np.array([1,2]),np.array([3,4]),np.array([4,3]),np.array([7.,6.]),3)
(p.sum()+n.sum()).backward();assert all(calls.values()),calls
assert model.time_encoder.basis_freq.grad is not None and torch.isfinite(model.time_encoder.basis_freq.grad).all()
# Single-query shape is explicitly supported by the port.
p,n=model.contrast(np.array([1]),np.array([3]),np.array([4]),np.array([7.]),3);assert p.shape==(1,)
for name,fun in originals.items():setattr(model_module,name,fun)
report={'status':'PASS','independent_temporal_cases':cases,'live_task_calls':calls,'single_query':'PASS','time_gradient':'PASS','all_padding_release_contract':'PASS'}
(Path(__file__).resolve().parent/'_check_l103_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
