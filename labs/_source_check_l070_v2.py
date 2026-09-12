"""L070 copied-weight operator check against immutable official TabM source."""
import hashlib,importlib.util,importlib.metadata,json,sys,urllib.request
from pathlib import Path
import torch
REV='28e47ae301c92ec37787dde1ce923a0793f405b4';URL=f'https://raw.githubusercontent.com/yandex-research/tabm/{REV}/tabm.py'
SHA='fc654af6a16bac53d893a8265c79d7af4ebddcb95ad0d600cc6b6bc6b7317ade'
def check(namespace=None):
 from relkit import checkpoint_l070_v2 as core
 s=vars(core) if namespace is None else namespace
 path=Path('/tmp/l070-official-tabm.py')
 if not path.exists():path.write_bytes(urllib.request.urlopen(URL).read())
 assert hashlib.sha256(path.read_bytes()).hexdigest()==SHA
 spec=importlib.util.spec_from_file_location('official_l070',path);official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
 torch.set_num_threads(1);maximum=0.;gradient=0.;cases=[]
 for din,width,depth,k in [(4,7,2,3),(8,64,3,8)]:
  torch.manual_seed(70);local=s['TabM'](din,width=width,depth=depth,k=k,dropout=0,arch='mini').double()
  ref=official.TabM.make(n_num_features=din,d_out=2,k=k,n_blocks=depth,d_block=width,dropout=0,arch_type='tabm-mini').double();pairs=[]
  for i,block in enumerate(local.blocks):
   rb=ref.backbone.blocks[i][0];pairs.extend([(block.weight,rb.weight,True),(block.bias,rb.bias,False)])
   assert block.bias.ndim==1 and block.S is None and (block.R is not None)==(i==0)
   assert float(block.weight.detach().abs().max())<=block.weight.shape[0]**-.5
   if i==0:pairs.append((block.R,ref.backbone.affine.weight,False))
  pairs.extend([(local.head.weight,ref.output.weight,False),(local.head.bias,ref.output.bias,False)])
  assert sum(p.numel() for p in local.parameters())==sum(p.numel() for p in ref.parameters())
  with torch.no_grad():
   for a,b,trans in pairs:b.copy_(a.T if trans else a)
  x=torch.randn(9,din,dtype=torch.float64,requires_grad=True);xx=x.detach().clone().requires_grad_(True);y=torch.arange(9)%2
  a,b=local(x),ref(xx);delta=float((a-b).detach().abs().max());maximum=max(maximum,delta)
  torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
  la=local.member_losses(x,y);lb=torch.nn.functional.cross_entropy(b.flatten(0,1),y.repeat_interleave(k));la.backward();lb.backward()
  torch.testing.assert_close(x.grad,xx.grad,atol=1e-12,rtol=1e-12)
  for a,b,trans in pairs:
   ga=a.grad.T if trans else a.grad;torch.testing.assert_close(ga,b.grad,atol=1e-12,rtol=1e-12);gradient=max(gradient,float((ga-b.grad).abs().max()))
  cases.append(dict(features=din,width=width,depth=depth,k=k,parameters=sum(p.numel() for p in local.parameters())))
 return dict(status='PASS',reference_import_dependency=dict(rtdl_num_embeddings=importlib.metadata.version('rtdl_num_embeddings')),official_revision=REV,url=URL,official_sha256=SHA,max_logit_difference=maximum,max_parameter_gradient_difference=gradient,cases=cases,scope='Numeric corrected mini; copied weights, every input/parameter gradient; dropout disabled. No optimizer RNG or full benchmark parity.')
if __name__=='__main__':
 r=check();(Path(__file__).parent/'_source_check_l070_v2_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
