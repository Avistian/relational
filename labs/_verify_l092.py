"""Behavioral oracles: endpoint multiplicity, attention axes, gradients, held-out labels."""
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent / "relkit"))
from han_l092 import metapath_reachability, neighbor_attention, semantic_fusion, masked_loss, HAN

def verify():
 torch.manual_seed(4)
 b=sp.csr_matrix([[1,1,0],[1,1,0],[0,1,1]],dtype=np.float32)
 a=metapath_reachability([b,b.T]);assert a[0,1]==1 and a[0,0]==1
 assert a.nnz==9,'Counts must collapse to distinct endpoints, including self'
 h=torch.tensor([[1.,0.],[0.,2.],[3.,1.]],requires_grad=True)
 edge=torch.tensor([[0,0,1,2],[0,2,1,2]])
 l=torch.tensor([0.,0.,0.],requires_grad=True);r=torch.tensor([0.,0.,0.],requires_grad=True)
 out=neighbor_attention(h,edge,l,r,0.,False)
 torch.testing.assert_close(out,torch.tensor([[2.,.5],[0.,2.],[3.,1.]]))
 out.sum().backward();assert h.grad is not None and r.grad.abs().sum()>0
 z=torch.tensor([[[1.,0.],[0.,2.]],[[3.,1.],[1.,0.]]],requires_grad=True)
 w=torch.eye(2,requires_grad=True);bias=torch.zeros(2,requires_grad=True);q=torch.tensor([1.,-1.],requires_grad=True)
 scores=torch.tanh(z)@q
 global_z,global_beta=semantic_fusion(z,w,bias,q,'paper_global')
 expected=scores.mean(0).softmax(0)
 torch.testing.assert_close(global_beta,expected.expand(2,-1))
 local_z,local_beta=semantic_fusion(z,w,bias,q,'release_node')
 torch.testing.assert_close(local_beta,scores.softmax(1));assert not torch.allclose(global_beta,local_beta)
 global_z.sum().backward();assert w.grad.abs().sum()>0 and q.grad.abs().sum()>0
 logits=torch.randn(3,3,requires_grad=True);y=torch.tensor([0,1,2]);idx=torch.tensor([0])
 loss=masked_loss(logits,y,idx);y[1:]=0;torch.testing.assert_close(loss,masked_loss(logits,y,idx));loss.backward();assert logits.grad[1:].abs().sum()==0
 # Independent dense masked-softmax forward and gradient oracle.
 n=5;edge=torch.tensor([[0,0,1,1,2,3,4],[0,2,0,1,2,3,4]])
 h=torch.randn(n,3,dtype=torch.float64,requires_grad=True);l=torch.randn(n,dtype=torch.float64,requires_grad=True);r=torch.randn(n,dtype=torch.float64,requires_grad=True)
 sparse=neighbor_attention(h,edge,l,r,0.,False)
 mask=torch.full((n,n),-torch.inf,dtype=h.dtype);mask[edge[0],edge[1]]=0
 dense=(torch.nn.functional.leaky_relu(l[:,None]+r[None,:],.2)+mask).softmax(1)@h
 torch.testing.assert_close(sparse,dense)
 gs=torch.autograd.grad(sparse.square().sum(),(h,l,r),retain_graph=True);gd=torch.autograd.grad(dense.square().sum(),(h,l,r))
 for x,y in zip(gs,gd):torch.testing.assert_close(x,y)
 return {'status':'PASS','reachability_endpoint_deduplication':'PASS','sparse_dense_outputs_and_gradients':'PASS','paper_global_vs_release_node':'PASS','heldout_label_gradient_isolation':'PASS'}
if __name__=='__main__':
 result=verify();Path(__file__).with_name('_verify_l092_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
