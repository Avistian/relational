"""Fresh isolated reference checks against the actual live L064 namespace."""
import hashlib,json,os,subprocess,sys,urllib.request,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
WHEEL_SHA='04e3bb989e9328d510ea4fccb6c6a36c8269630685d460af4df5eb268206bf21'
WHEEL_URL='https://files.pythonhosted.org/packages/35/f8/b0e7ff83484acfbe670b2f56c28c6bcf09ae84284e624cd45755a8b7073c/tabpfn-2.0.9-py3-none-any.whl'
CHECKS={
'group_features':'''# CHECK — identity-coded rows expose an axis bug that shapes alone miss
x=torch.tensor([[[0.,1.,2.],[10.,11.,12.]]])
g=group_features(x)
assert g.shape==(1,2,2,2)
assert torch.equal(g[0,1],torch.tensor([[10.,11.],[12.,0.]])), 'Pad the feature axis before grouping; preserve rows'
print('Group identities and odd-feature padding pass')''',
'encode_targets':'''# CHECK — the query placeholder is mean-imputed then ranked, with its own missing flag
encoded=encode_targets(torch.tensor([[0.,0.,1.]]),5)
assert torch.equal(encoded[0],torch.tensor([[0.,0.],[0.,0.],[1.,0.],[1.,-2.],[1.,-2.]])), 'Query target uses context mean rank plus -2 flag; never true query labels'
print('Observed targets and unknown-target semantics pass')''',
'attention_mix':'''# CHECK — synthetic scores 0,1 normalize across the two senders
q=torch.tensor([[[[1.,0.]]]]);k=torch.tensor([[[[0.,0.],[2**.5,0.]]]]);v=torch.tensor([[[[1.,0.],[0.,2.]]]])
assert torch.allclose(attention_mix(q,k,v)[0,0,0],torch.tensor([.26894142,1.46211716]),atol=1e-6), 'Softmax over senders, scaled by sqrt(head width)'
print('Rectangular attention arithmetic passes')''',
'row_attention':'''# CHECK — context uses every KV head; queries reuse first KV head
calls=[]
def probe(receiver,sender,first_kv=False):
    calls.append((receiver.clone(),sender.clone(),first_kv))
    return receiver+sender.mean(-2,keepdim=True)
h=torch.arange(1*5*3*4.).reshape(1,5,3,4)
z=row_attention(h,3,probe)
assert z.shape==h.shape and len(calls)==2
assert calls[0][0].shape==(1,3,3,4) and calls[1][0].shape==(1,3,2,4)
assert not calls[0][2] and calls[1][2], 'Multi-query K/V sharing is only on query receivers'
assert torch.equal(calls[1][1],h[:,:3].transpose(1,2)), 'Only labeled context can send'
print('Both row routes and first-head KV flag pass')''',
'postnorm_update':'''# CHECK — non-affine post-addition norm, no trainable scale or shift
h=torch.tensor([[[[1.,2.,4.]]]]);u=torch.tensor([[[[2.,0.,1.]]]])
expected=torch.nn.functional.layer_norm(h+u,(3,),eps=1e-5)
assert torch.allclose(postnorm_update(h,u),expected)
assert not torch.allclose(expected,torch.nn.functional.layer_norm(h,(3,))+u), 'Residual belongs inside norm'
print('Residual and postnorm pass')'''}

def reference_result(checkpoint):
 cache=ROOT/'data/cache/l064-source';cache.mkdir(parents=True,exist_ok=True);wheel=cache/'tabpfn.whl'
 if not wheel.exists():urllib.request.urlretrieve(WHEEL_URL,wheel)
 if hashlib.sha256(wheel.read_bytes()).hexdigest()!=WHEEL_SHA:raise ValueError('Reference wheel checksum')
 official=cache/'official';zipfile.ZipFile(wheel).extractall(official)
 if not (official/'sklearn').exists():
  subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',str(official),'scikit-learn==1.6.1'],check=True)
 output=cache/'fresh-reference.pt';env=dict(os.environ,PYTHONPATH=str(official))
 subprocess.run([sys.executable,str(ROOT/'_reference_l064_v2.py'),str(checkpoint),str(output)],env=env,check=True)
 return torch.load(output,map_location='cpu',weights_only=False)

def map_key(key):
 return key.replace('encoder.5.layer.','x_encoder.').replace('y_encoder.2.layer.','y_encoder.').replace('feature_positional_embedding_embeddings.','position.').replace('decoder_dict.standard.','head.').replace('transformer_encoder.layers.','blocks.').replace('self_attn_between_features._w_','feature.').replace('self_attn_between_items._w_','row.').replace('mlp.linear1.','ff1.').replace('mlp.linear2.','ff2.')

def check(namespace=None,save=True):
 if namespace is None:
  from relkit import tabpfn_l064_v2 as core
  namespace=vars(core)
 s=namespace;torch.set_num_threads(1);path=s['ensure_checkpoint'](ROOT);reference=reference_result(path)
 model,_=s['load_pretrained'](path,s['TabPFNv2'](**s['MODEL_CONFIG']));model.double()
 full=reference['full'];x=full['x'].clone().requires_grad_();a=model(x,full['y']);a.square().mean().backward()
 out_delta=float((a.detach()-full['out']).abs().max());assert torch.allclose(a,full['out'],atol=2e-10,rtol=1e-10),out_delta
 input_delta=float((x.grad-full['xgrad']).abs().max());assert torch.allclose(x.grad,full['xgrad'],atol=1e-9,rtol=1e-8)
 local_params=dict(model.named_parameters());gradient_deltas=[]
 for key,g in full['grads'].items():
  local=local_params[map_key(key)].grad;assert local is not None and g is not None
  assert torch.allclose(local,g,atol=1e-8,rtol=1e-7),(key,float((local-g).abs().max()))
  gradient_deltas.append(float((local-g).abs().max()))
 model.float();model.zero_grad();cases=[]
 for case in reference['wrappers']:
  x=case['x'];y=case['y'];p,trace=s['predict_numeric'](model,x[:24],y,x[24:]);delta=float(np.max(abs(p.numpy()-case['p'])))
  assert np.allclose(p.numpy(),case['p'],atol=4e-5,rtol=4e-5),(case['kind'],delta)
  cases.append(dict(case=case['kind'],probability_max_delta=delta))
 # Default wrapper end-to-end bridge: its actual numeric transformations are
 # peripheral provided code; every network output is recomputed by live code.
 default=reference['default'];outputs=[];raw_deltas=[]
 assert not default['average_before_softmax'] and not default['balance_probabilities']
 for call,permutation in zip(default['calls'],default['permutations']):
  with torch.no_grad():logits=model(call['x'].transpose(0,1),call['y'].reshape(call['n'],-1).T,seed=default['model_seed'],outlier_sigma=default['outlier_sigma'])[0,:,:default['classes']]
  outputs.append((logits[:,permutation]/default['temperature']).softmax(-1))
 p=torch.stack(outputs).mean(0);p=p/p.sum(-1,keepdim=True);delta=float(np.max(abs(p.numpy()-default['p'])))
 assert np.allclose(p.numpy(),default['p'],atol=6e-5,rtol=6e-5),('default bridge',delta)
 # Context-label sensitivity and whole-wrapper query coupling are different tests.
 x=np.column_stack([np.linspace(-2,2,12),np.ones(12)]).astype('float32');x[1,1]=np.nan;y=(x[:,0]>0).astype(int)
 q=np.array([[.3,1.]],dtype='float32');added=np.array([[.3,1.],[0.,2.]],dtype='float32')
 alone,_=s['predict_numeric'](model,x,y,q);joined,_=s['predict_numeric'](model,x,y,added)
 coupling=float((alone-joined[:1]).abs().max());assert coupling>1e-3
 assert np.allclose(alone.numpy(),reference['coupling']['alone'],atol=4e-5,rtol=4e-5)
 assert np.allclose(joined[:1].numpy(),reference['coupling']['joined'],atol=4e-5,rtol=4e-5)
 clean=x.copy();clean[1,1]=1
 clean_a,_=s['predict_numeric'](model,clean,y,q);clean_b,_=s['predict_numeric'](model,clean,y,added)
 clean_delta=float((clean_a-clean_b[:1]).abs().max());assert clean_delta<3e-5
 # Frozen encoded tokens isolate attention from the preceding count operation.
 torch.manual_seed(71);h=torch.randn(1,8,4,192);block=model.blocks[0]
 with torch.no_grad():
  a=block(h,6);changed=h.clone();changed[:,-1]*=20;b=block(changed,6)
 mask_delta=float((a[:,:7]-b[:,:7]).abs().max());assert mask_delta<1e-5
 result=dict(status='PASS',wheel_sha256=WHEEL_SHA,checkpoint_sha256=s['CHECKPOINT_SHA'],parameter_count=sum(p.numel() for p in model.parameters()),parameter_tensors=len(local_params),full_float64_logit_max_delta=out_delta,input_gradient_max_delta=input_delta,parameter_gradient_tensors=len(gradient_deltas),parameter_gradient_max_delta=max(gradient_deltas),whole_numeric_wrapper_cases=cases,default_four_view_wrapper_bridge=dict(probability_max_delta=delta,views=len(outputs),model_seed=default['model_seed'],temperature=default['temperature'],outlier_sigma=default['outlier_sigma'],boundary='Official transformed tables captured; live model recomputes every layer, outlier transform and output; official preprocessing algorithms not independently reimplemented'),counterexample=dict(context_x=np.nan_to_num(x,nan=0).tolist(),context_missing_mask=np.isnan(x).tolist(),context_y=y.tolist(),queries=added.tolist(),alone=alone.tolist(),joined=joined[:1].tolist(),probability_delta=coupling,finite_constant_control_delta=clean_delta),fixed_token_mask_max_delta=mask_delta)
 if save:(ROOT/'_check_l064_v2_results.json').write_text(json.dumps(result,indent=2)+'\n')
 return result
if __name__=='__main__':print(json.dumps(check(),indent=2))
