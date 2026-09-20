"""Behavioral contract: numeric traces, cross-relation normalization, and source parity."""
import importlib.util,json,math,sys,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs'
sys.path.insert(0,str(LAB))
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'relkit'))
from hgt_l093 import receiver_softmax,relation_heads,temporal_basis,HGTLayer

torch.set_num_threads(2);torch.manual_seed(7)
z=torch.tensor([[0.,math.log(2)],[math.log(2),0.],[0.,0.]],requires_grad=True)
dst=torch.tensor([2,2,3]);a=receiver_softmax(z,dst,5)
torch.testing.assert_close(a,torch.tensor([[1/3,2/3],[2/3,1/3],[1.,1.]]))
(a*torch.arange(6.).reshape(3,2)).sum().backward();assert z.grad[:2].abs().sum()>0
q=torch.tensor([[[1.,0.]],[[1.,0.]]]);k=torch.tensor([[[0.,1.]],[[math.sqrt(2)*math.log(2),0.]]]);v=torch.tensor([[[2.,0.]],[[6.,0.]]]);w=torch.eye(2).reshape(1,1,2,2).expand(2,1,2,2);p=torch.ones(2,1)
s,m=relation_heads(q,k,v,w,w,p);torch.testing.assert_close(s[:,0],torch.tensor([0.,math.log(2)]));torch.testing.assert_close(m,v)
x=temporal_basis(torch.tensor([0.,1.]),4,'release');torch.testing.assert_close(x[0],torch.tensor([0.,.5,0.,.5]));assert x.shape==(2,4)
# Source oracle runs the authors' actual class through installed PyG, without replacing message passing.
spec=importlib.util.spec_from_file_location('hgt_source',LAB/'sources/hgt-l093/OAG/pyHGT/conv.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
records=[]
for rte in [False,True]:
 for norm in [False,True]:
  ref=mod.HGTConv(8,8,3,4,2,dropout=0.,use_norm=norm,use_RTE=rte)
  ours=HGTLayer(8,3,4,2,dropout=0.,use_norm=norm,use_rte=rte)
  ours.load_state_dict(ref.state_dict(),strict=True)
  xx=torch.randn(7,8,requires_grad=True);yy=xx.detach().clone().requires_grad_()
  nt=torch.tensor([0,1,2,0,1,2,2]);ei=torch.tensor([[0,1,2,3,4,5,0,2,6],[2,2,2,1,1,0,0,3,6]]);er=torch.tensor([0,1,2,0,1,2,3,3,3]);et=torch.tensor([120,121,119,122,120,120,120,123,120])
  u=ref(xx,nt,ei,er,et);v=ours(yy,nt,ei,er,et)
  torch.testing.assert_close(u,v,atol=2e-6,rtol=2e-5);torch.testing.assert_close(ref.att,ours.att,atol=1e-6,rtol=1e-5)
  target=torch.randn_like(u);(u*target).sum().backward();(v*target).sum().backward()
  torch.testing.assert_close(xx.grad,yy.grad,atol=3e-6,rtol=3e-5)
  for (rn,rp),(on,op) in zip(ref.named_parameters(),ours.named_parameters()):
   assert rn==on,(rn,on)
   if rp.grad is not None:torch.testing.assert_close(rp.grad,op.grad,atol=4e-6,rtol=4e-5)
  if rte:assert ref.emb.emb.weight.requires_grad and ours.emb.emb.weight.grad.abs().sum()>0
  records.append({'rte':rte,'norm':norm,'max_output_error':float((u-v).abs().max().detach())})
result={'status':'PASS','numeric_trace':'PASS','all_incoming_relations_share_softmax':'PASS','upstream_output_attention_input_and_parameter_gradients':records,'historical_paper_parity':'NOT_ESTABLISHED'}
(LAB/'_verify_l093_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
