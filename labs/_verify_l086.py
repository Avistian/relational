"""Independent arithmetic/gradient oracle and real PyG batching checks."""
import json
from pathlib import Path
import torch
from torch_geometric.nn import GCNConv
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"relkit"))
from pyg_l086 import *
from gcn_l082 import normalized_support,load_cora,propagate
import numpy as np
root=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.manual_seed(86)
d=toy_data();conv=TraceGCN(1,2).double();x=d.x.double().requires_grad_()
a=np.zeros((4,4));a[d.edge_index[1],d.edge_index[0]]=1
s=normalized_support(a).to(torch.float64)
# Use dense independent oracle recomputed in double precision.
t=torch.tensor(a,dtype=torch.float64)+torch.eye(4,dtype=torch.float64);q=t.sum(1).rsqrt();s=q[:,None]*t*q[None,:]
y=conv(x,d.edge_index);oracle=s@x@conv.weight
torch.testing.assert_close(y,oracle,rtol=1e-12,atol=1e-12)
g1=torch.autograd.grad(y.square().sum(),(x,conv.weight),retain_graph=True)
g2=torch.autograd.grad(oracle.square().sum(),(x,conv.weight))
for left,right in zip(g1,g2):torch.testing.assert_close(left,right,rtol=1e-12,atol=1e-12)
# Independent built-in operator, with explicit matching bias and weight.
builtin=GCNConv(1,2,bias=False).double()
with torch.no_grad():builtin.lin.weight.copy_(conv.weight.T)
torch.testing.assert_close(y,builtin(x,d.edge_index),rtol=1e-12,atol=1e-12)
# Permutation equivariance, isolated node, duplicate existing self-loop replacement.
p=torch.tensor([2,0,3,1]);inv=torch.argsort(p)
torch.testing.assert_close(conv(x[p],inv[d.edge_index]),y[p])
loops=torch.stack([torch.arange(4),torch.arange(4)])
torch.testing.assert_close(conv(x,torch.cat([d.edge_index,loops],1)),y)
torch.testing.assert_close(y[3],x[3]@conv.weight)
# Directed routing oracle with supplied coefficients bypassing GCN normalization.
z=conv.propagate(torch.tensor([[0],[1]]),x=x,norm=torch.ones(1,dtype=x.dtype),size=(4,4))
torch.testing.assert_close(z,torch.tensor([[0.],[2.],[0.],[0.]],dtype=x.dtype))
# Block diagonal graph batching cannot create cross-graph messages.
batch=Batch.from_data_list([d,d]);out=TraceGCN(1,2);torch.testing.assert_close(out(batch.x,batch.edge_index),out(d.x,d.edge_index).repeat(2,1))
assert not (batch.batch[batch.edge_index[0]]!=batch.batch[batch.edge_index[1]]).any()
# Full Cora output parity with L082, identical weights, evaluation only.
loaded=load_cora(root);data=as_data(loaded);c=TraceGCN(1433,16)
with torch.no_grad():
 actual=c(data.x,data.edge_index);expected=propagate(loaded[1],loaded[0],c.weight)
 torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
error=float((actual-expected).abs().max())
# Heterogeneous IDs belong to separate type namespaces.
h=HeteroData();h['author'].x=torch.ones(2,3);h['paper'].x=torch.ones(3,4)
h['author','writes','paper'].edge_index=torch.tensor([[0,1],[1,2]])
assert h.validate(raise_on_error=True)
result={'status':'PASS','double_output_and_gradients':'PASS','directed_routing':'PASS','permutation_and_isolation':'PASS','block_diagonal_batch':'PASS','heterodata':'PASS','full_cora_l082_max_error':error,'neighbor_loader':loader_check()}
(root/'_verify_l086_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
