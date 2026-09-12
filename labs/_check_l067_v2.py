"""Independent official LoCalPFN full forward, parameter gradients and AdamW parity."""
import hashlib,importlib.util,json,copy,urllib.request
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
COMMIT='ff8803c57cd277380b2444f0f5ed4856b46f47f5'

def reference_model(root=ROOT):
 p=Path(root)/'data/cache/l067-source/official/pfn.py'
 manifest=json.loads((Path(root)/'_sources_l067_v2.json').read_text());expected=manifest['files']['pfn.py']['sha256']
 if not p.exists():
  p.parent.mkdir(parents=True,exist_ok=True);urllib.request.urlretrieve(f'https://raw.githubusercontent.com/layer6ai-labs/LoCalPFN/{COMMIT}/pfn.py',p)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
 spec=importlib.util.spec_from_file_location('official_l067_pfn',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 from relkit import tabpfn_l062_v2 as backbone
 state,_,cfg=torch.load(backbone.ensure_checkpoint(root),map_location='cpu',weights_only=False)
 model=module.PFN(dropout=cfg['dropout'],embedding_normalization=False,n_out=cfg['max_num_classes'],nhead=cfg['nhead'],nhid=cfg['emsize']*cfg['nhid_factor'],ninp=cfg['emsize'],nlayers=cfg['nlayers'],norm_first=False,num_features=cfg['num_features'])
 mapped={k.removeprefix('module.').replace('layers.',''):v for k,v in state.items() if not k.endswith('criterion.weight')}
 model.load_state_dict(mapped,strict=True);return model.eval()

def mapped_name(name):
 key=name.replace('blocks.','transformer_encoder.').replace('qkv.weight','self_attn.in_proj_weight').replace('qkv.bias','self_attn.in_proj_bias').replace('.out.','.self_attn.out_proj.').replace('.ff1.','.linear1.').replace('.ff2.','.linear2.')
 if key.startswith('x_encoder.'):key='encoder.'+key[len('x_encoder.'):]
 if key.startswith('head.'):key='decoder.'+key[len('head.'):]
 return key

def check(namespace=None,save=True):
 from relkit import localpfn_l067_v2 as core
 s=vars(core) if namespace is None else namespace;torch.set_num_threads(1);torch.manual_seed(670)
 local,_=s['load_pretrained'](s['ensure_checkpoint'](ROOT));official=reference_model();identity=s['kernel_identity'](s,ROOT)
 report=dict(status='PASS',kernel_identity=identity,initial_weights_sha256=s['model_digest'](local),initial_runtime_sha256=s['model_runtime_identity'](local),cases=[])
 # Complete pretrained forward AND parameter gradients through actual local wrapper.
 for features,classes,batch,context,queries in [(4,2,2,9,3),(7,3,1,11,4),(8,2,2,8,1)]:
  x=torch.randn(batch,context+queries,features);y=torch.arange(batch*context).reshape(batch,context)%classes
  # Third case has single-class contexts; global classes must stay 2.
  if queries==1:y[:]=0
  local.zero_grad();official.zero_grad()
  a=local(s['local_normalize'](x,context),y.float())
  full=torch.nn.functional.pad(x,(0,100-features)).transpose(0,1).contiguous()
  yy=torch.cat([y,torch.full((batch,queries),99)],1).T.float()
  b=official(full,yy,context,normalization=True,outlier_clipping=False,nan_replacement=False,used_features=features).transpose(0,1)
  delta=float((a-b).detach().abs().max());torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-4)
  target=torch.arange(batch*queries).reshape(batch,queries)%classes
  loss=s['query_loss'](a,target,classes);other=torch.nn.functional.cross_entropy(b[...,:classes].reshape(-1,classes),target.reshape(-1))
  loss.backward();other.backward();deltas=[]
  for name,p in local.named_parameters():
   ref=dict(official.named_parameters())[mapped_name(name)];torch.testing.assert_close(p.grad,ref.grad,atol=2e-4,rtol=2e-3);deltas.append(float((p.grad-ref.grad).abs().max()))
  report['cases'].append(dict(features=features,classes=classes,context=context,queries=queries,batch=batch,logit_max_delta=delta,loss=float(loss.detach()),gradient_max_delta=max(deltas),gradient_tensors=len(deltas)))
 # Constant float32 real-value regression: source sum/count can amplify rounding.
 x=torch.zeros(4,33,8);x[:,:,3]=-1.3079352378845215
 actual=s['local_normalize'](x,32)
 spec=importlib.util.spec_from_file_location('source_moments',ROOT/'data/cache/l067-source/official/pfn.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 expected=(mod.normalize_data(torch.nn.functional.pad(x,(0,92)).transpose(0,1).contiguous(),32)/(8/100)).transpose(0,1)
 torch.testing.assert_close(actual,expected,atol=0,rtol=0)
 report['constant_feature_source_value']=float(actual[0,0,3]);report['constant_feature_max_delta']=float((actual-expected).abs().max())
 # One complete AdamW update: use double precision to avoid sign noise around tiny gradients.
 local.double();official.double();x=torch.randn(1,13,5,dtype=torch.float64);y=(torch.arange(9)%2)[None];target=(torch.arange(4)%2)[None]
 ol=torch.optim.AdamW(local.parameters(),lr=.01,weight_decay=.01);oo=torch.optim.AdamW(official.parameters(),lr=.01,weight_decay=.01)
 ol.zero_grad();oo.zero_grad()
 a=local(s['local_normalize'](x,9),y.double());b=official(torch.nn.functional.pad(x,(0,95)).transpose(0,1),torch.cat([y,torch.zeros(1,4)],1).T.double(),9,True,False,False,5).transpose(0,1)
 s['query_loss'](a,target,2).backward();torch.nn.functional.cross_entropy(b[...,:2].reshape(-1,2),target.reshape(-1)).backward();ol.step();oo.step()
 deltas=[]
 for name,p in local.named_parameters():
  ref=dict(official.named_parameters())[mapped_name(name)];torch.testing.assert_close(p,ref,atol=3e-8,rtol=3e-8);deltas.append(float((p-ref).detach().abs().max()))
 report['one_adamw_step_parameter_max_delta']=max(deltas)
 # Operational checks of identities, ties, duplicate values and one-class query contracts.
 x=np.array([[0.,0.],[0.,0.],[2.,0.],[3.,0.]])
 assert s['neighbor_ids'](x,x[:1],2,[10,11,12,13],[10]).tolist()==[[1,2]]
 c,q=s['episode_indices'](np.array([[1,2,3,4],[4,3,2,1]]),2,np.random.default_rng(4))
 assert all(not set(a)&set(b) for a,b in zip(c,q))
 assert s['validation_choice']([.7,.7,.6])==0 and s['validation_choice']([.7,.8,.8])==1
 report['checker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();report['source_manifest_sha256']=hashlib.sha256((ROOT/'_sources_l067_v2.json').read_bytes()).hexdigest()
 assert s['kernel_identity'](s,ROOT)['sha256']==identity['sha256']
 if save:(ROOT/'_check_l067_v2_results.json').write_text(json.dumps(report,indent=2)+'\n')
 return report
if __name__=='__main__':print(json.dumps(check(),indent=2))
