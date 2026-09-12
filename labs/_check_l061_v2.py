"""L061 analytic, information-flow and official copied-weight validation."""
import json,math,hashlib,importlib.util,sys,types
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import quad
ROOT=Path(__file__).resolve().parent
CHECKS={
'sample_gp':'''# CHECK — one jointly sampled GP, not independent row labels
_generator=torch.Generator().manual_seed(17)
_x,_y=sample_gp(12000,2,1,_generator)
assert _x.shape==(12000,2,1) and _y.shape==(12000,2)
_expected=rbf_kernel(_x,_x,.6)[:,0,1].mean().item()
assert abs((_y[:,0]*_y[:,1]).mean().item()-_expected)<.04, 'Context and query must share a GP draw'
assert abs(_y.var().item()-1.0001)<.04, 'Noise is variance added to K, not a replacement for task covariance'
''',
'gp_posterior':'''# CHECK — one observation admits an independent scalar calculation
_x=torch.tensor([[[0.],[1.]]],dtype=torch.float64);_y=torch.tensor([[2.]],dtype=torch.float64)
_mu,_var=gp_posterior(_x,_y,1)
_k=math.exp(-1/(2*.6**2))
assert abs(_mu.item()-2*_k/1.0001)<1e-10, 'Solve Kcc against yc before multiplying Kqc'
assert abs(_var.item()-(1.0001-_k**2/1.0001))<1e-10, 'Predict noisy y: add query noise once'
_mu0,_v0=gp_posterior(_x,_y[:,:0],0)
assert torch.equal(_mu0,torch.zeros_like(_mu0)) and torch.allclose(_v0,torch.full_like(_v0,1.0001))
''',
'context_attention_mask':'''# CHECK — rows receive, columns send; the original code keeps query self edges
_mask=context_attention_mask(4,2)
_expected=torch.tensor([[1,1,0,0],[1,1,0,0],[1,1,1,0],[1,1,0,1]],dtype=torch.bool)
assert torch.equal(torch.isfinite(_mask),_expected), 'Query self is allowed, another query is blocked'
assert torch.equal(torch.isfinite(context_attention_mask(3,0)),torch.eye(3,dtype=torch.bool)), 'Empty context uses unlabeled self only'
''',
'attention_mix':'''# CHECK — masked scaled scores, row softmax and value aggregation
_q=torch.zeros(1,1,3,1);_k=_q.clone();_v=torch.tensor([[[[2.],[6.],[90.]]]])
_out=attention_mix(_q,_k,_v,context_attention_mask(3,2))
assert torch.allclose(_out.flatten(),torch.tensor([4.,4.,98/3])), 'Context reads two values; query reads them and itself'
''',
'riemann_nll':'''# CHECK — density is mass / width; tails integrate to their assigned mass
_b=torch.tensor([-3.,-1.,0.,2.,5.],dtype=torch.float64)
_logits=torch.log(torch.tensor([[.1,.2,.3,.4]],dtype=torch.float64))
assert abs(riemann_nll(_logits,torch.tensor([1.],dtype=torch.float64),_b).item()+math.log(.3/2))<1e-12
_tail=riemann_nll(_logits,torch.tensor([12.],dtype=torch.float64),_b)
assert torch.isfinite(_tail).all(), 'Unbounded GP targets need nonzero tail density'
assert torch.allclose(riemann_nll(_logits+100,torch.tensor([1.],dtype=torch.float64),_b),riemann_nll(_logits,torch.tensor([1.],dtype=torch.float64),_b))
'''}

def check(namespace=None,save=True):
 path=ROOT/'relkit/pfn_l061_v2.py'
 assert path.exists(), 'The row-token PFN mechanism has not yet been implemented'
 from relkit import pfn_l061_v2 as m
 if namespace is not None:m=types.SimpleNamespace(**namespace)
 env=dict(vars(m));env.update(math=math,torch=torch)
 torch.set_num_threads(1)
 for name,source in CHECKS.items():exec(source,env)
 result={'status':'PASS','checks':list(CHECKS)}
 # Load original modules unchanged; isolate the historical top-level utils name.
 from _fetch_l061_v2 import fetch
 folder,sources=fetch()
 def load(name,path):
  spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
 previous=sys.modules.get('utils');sys.modules['utils']=load('l061_official_utils',folder/'utils.py')
 try:official=load('l061_official_transformer',folder/'transformer.py')
 finally:
  if previous is None:sys.modules.pop('utils',None)
  else:sys.modules['utils']=previous
 bar=load('l061_official_bar',folder/'bar_distribution.py')
 torch.manual_seed(88)
 ours=m.RowPFN(features=2,width=16,heads=4,hidden=32,layers=2,bins=5).double().eval()
 ref=official.TransformerModel(torch.nn.Linear(2,16),5,16,4,32,2,y_encoder=torch.nn.Linear(1,16)).double().eval()
 # Randomize every branch, so zero residual initialization cannot hide mask errors.
 for parameter in ref.parameters():torch.nn.init.normal_(parameter,std=.2)
 ours.x_encoder.load_state_dict(ref.encoder.state_dict());ours.y_encoder.load_state_dict(ref.y_encoder.state_dict());ours.head.load_state_dict(ref.decoder.state_dict())
 for a,b in zip(ours.blocks,ref.transformer_encoder.layers):
  a.qkv.weight.data.copy_(b.self_attn.in_proj_weight);a.qkv.bias.data.copy_(b.self_attn.in_proj_bias)
  for target,source in [(a.out,b.self_attn.out_proj),(a.ff1,b.linear1),(a.ff2,b.linear2),(a.norm1,b.norm1),(a.norm2,b.norm2)]:target.load_state_dict(source.state_dict())
 errors=[];grad_errors=[];parameter_errors=[]
 for n in [0,1,3]:
  x=torch.randn(2,6,2,dtype=torch.float64,requires_grad=True);y=torch.randn(2,6,dtype=torch.float64)
  mask=m.context_attention_mask(6,n).double()
  actual=ours(x,y[:,:n]);expected=ref((x.transpose(0,1),y.T),single_eval_pos=n,src_mask=mask).transpose(0,1)
  errors.append(float((actual-expected).abs().max().detach()));assert torch.allclose(actual,expected,atol=1e-10,rtol=1e-10)
  if n==3:
   pairs=list(zip(ours.x_encoder.parameters(),ref.encoder.parameters()))+list(zip(ours.y_encoder.parameters(),ref.y_encoder.parameters()))+list(zip(ours.head.parameters(),ref.decoder.parameters()))
   for a,b in zip(ours.blocks,ref.transformer_encoder.layers):
    pairs += [(a.qkv.weight,b.self_attn.in_proj_weight),(a.qkv.bias,b.self_attn.in_proj_bias)]
    for target,source in [(a.out,b.self_attn.out_proj),(a.ff1,b.linear1),(a.ff2,b.linear2),(a.norm1,b.norm1),(a.norm2,b.norm2)]:pairs+=list(zip(target.parameters(),source.parameters()))
   left=torch.autograd.grad(actual.square().sum(),[a for a,b in pairs],retain_graph=True)
   right=torch.autograd.grad(expected.square().sum(),[b for a,b in pairs],retain_graph=True)
   parameter_errors=[float((a-b).abs().max()) for a,b in zip(left,right)]
   assert max(parameter_errors)<1e-10
  grad_a=torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0];grad_b=torch.autograd.grad(expected.square().sum(),x)[0]
  grad_errors.append(float((grad_a-grad_b).abs().max()));assert torch.allclose(grad_a,grad_b,atol=1e-10,rtol=1e-10)
  assert torch.equal(m.context_attention_mask(6,n).double(),ref.generate_D_q_matrix(6,6-n).double())
 x=torch.rand(2,6,2,dtype=torch.float64);y=torch.randn(2,3,dtype=torch.float64)
 baseline=ours(x,y);order=torch.tensor([2,0,1]);permuted=torch.cat([x[:,:3][:,order],x[:,3:]],1)
 assert torch.allclose(ours(permuted,y[:,order]),baseline,atol=1e-10), 'Context permutation changed query predictions'
 single=ours(torch.cat([x[:,:3],x[:,3:4]],1),y)
 assert torch.allclose(single,baseline[:,:1],atol=1e-10), 'Query packing changed another query prediction'
 changed=x.clone();changed[:,4:]+=100
 assert torch.allclose(ours(changed,y)[:,:1],baseline[:,:1],atol=1e-10)
 assert not torch.allclose(ours(x,y+2),baseline,atol=1e-7), 'Context label path is dead'
 borders=torch.tensor([-3.,-1.,0.,2.,5.],dtype=torch.float64)
 logits=torch.randn(11,4,dtype=torch.float64,requires_grad=True)
 target=torch.tensor([-12.,-3.,-2.,-1.,-.5,0.,1.,2.,4.,5.,12.],dtype=torch.float64)
 a=m.riemann_nll(logits,target,borders);b=bar.FullSupportBarDistribution(borders)(logits,target)
 density_error=float((a-b).abs().max().detach());assert torch.allclose(a,b,atol=1e-7)
 ga=torch.autograd.grad(a.sum(),logits,retain_graph=True)[0];gb=torch.autograd.grad(b.sum(),logits)[0]
 assert torch.allclose(ga,gb,atol=1e-12)
 probs=torch.tensor([[.1,.2,.3,.4]],dtype=torch.float64).log()
 integral=sum(quad(lambda y: math.exp(-m.riemann_nll(probs,torch.tensor([y],dtype=torch.float64),borders).item()),a,b,epsabs=1e-8)[0] for a,b in [(-np.inf,-1),(-1,0),(0,2),(2,np.inf)])
 assert abs(integral-1)<1e-8
 assert torch.allclose(m.riemann_mean(probs,borders),bar.FullSupportBarDistribution(borders).mean(probs))
 for point in [-4.,-.5,1.,7.]:
  integral_to=quad(lambda y:math.exp(-m.riemann_nll(probs,torch.tensor([y],dtype=torch.float64),borders).item()),-np.inf,point,epsabs=1e-7,limit=200)[0]
  assert abs(m.riemann_cdf(probs,torch.tensor([point],dtype=torch.float64),borders).item()-integral_to)<2e-6
 ours.float();ref.float();xf=torch.rand(2,6,2);yf=torch.randn(2,6)
 af=ours(xf,yf[:,:3]);bf=ref((xf.transpose(0,1),yf.T),single_eval_pos=3).transpose(0,1)
 production_error=float((af-bf).abs().max().detach());assert torch.allclose(af,bf,atol=2e-6,rtol=2e-6)
 result.update(float32_implicit_mask_max_forward_error=production_error,official_sources=sources,max_forward_error=max(errors),max_input_gradient_error=max(grad_errors),max_parameter_gradient_error=max(parameter_errors),parameter_tensors_checked=len(parameter_errors),max_density_error=density_error,parity_adapter='float64 test passes same-valued mask cast to float64 explicitly; original default creates float32 mask; original tail median computed float32 gives submicro density difference',density_integral=integral,information_flow='context permutation, query independence, label sensitivity, zero context PASS')
 if save:(ROOT/'_check_l061_v2_results.json').write_text(json.dumps(result,indent=2)+'\n')
 return result
if __name__=='__main__':print(json.dumps(check(),indent=2))
