"""Pinned released-code validation of the actual live L062 namespace."""
import hashlib,json,os,sys,typing,urllib.request,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
WHEEL_URL='https://files.pythonhosted.org/packages/0e/e9/b95d9970c0a55c8bc41bb8b653c21b7cc034506cca4d6f6929d97f04f41e/tabpfn-0.1.11-py3-none-any.whl'
WHEEL_SHA='d7699467049cf6e3950121777a5ee6aa625cb9066447a7888f8ebf8937d3858a'

def official_source():
    cache=ROOT/'data/cache/l062-source';cache.mkdir(parents=True,exist_ok=True);p=cache/'tabpfn.whl'
    if not p.exists():urllib.request.urlretrieve(WHEEL_URL,p)
    assert hashlib.sha256(p.read_bytes()).hexdigest()==WHEEL_SHA
    # Extract verified wheel each time so modified cached source is not silently trusted.
    zipfile.ZipFile(p).extractall(cache/'official');sys.path.insert(0,str(cache/'official'))
    import torch.nn.modules.transformer as module
    if not hasattr(module,'Optional'):module.Optional=typing.Optional
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    from tabpfn.scripts.model_builder import load_model_only_inference
    from tabpfn.scripts.transformer_prediction_interface import transformer_predict
    from tabpfn.utils import normalize_data,remove_outliers
    return load_model_only_inference,transformer_predict,normalize_data,remove_outliers

CHECKS={
'normalize_context':'''# CHECK — held-out extremes cannot move context statistics
fixture=torch.tensor([[1.,2.],[3.,6.],[101.,-100.]])
z=normalize_context(fixture,2)
assert torch.allclose(z[0],torch.tensor([-.7071063,-.7071065]),atol=1e-6), 'Use sample std and context rows only'
changed=fixture.clone();changed[-1]*=100
assert torch.equal(z[:2],normalize_context(changed,2)[:2]), 'Query-fitted statistics leak through preprocessing'
assert z.abs().max()<=100
print('Context-only sample moments and clamp pass')''',
'scale_pad':'''# CHECK — fixed input width, factor K/k and exact zero tail
z=scale_pad(torch.tensor([[1.,-2.]]),6)
assert torch.equal(z,torch.tensor([[3.,-6.,0.,0.,0.,0.]])), 'Use K/k, then pad; sqrt scaling changes the pretrained input'
print('Flexible input scaling passes')''',
'attention_mix':'''# CHECK — analytic two-sender fixture and receiver independence
q=torch.tensor([[[[1.,0.],[0.,1.]]]]);k=torch.tensor([[[[0.,0.],[2**.5,0.]]]]);v=torch.tensor([[[[1.,0.],[0.,2.]]]])
a=attention_mix(q,k,v)
assert torch.allclose(a[0,0,0],torch.tensor([.26894142,1.46211716]),atol=1e-6), 'Scale by head width and normalize over senders'
assert torch.allclose(a[:,:,0],attention_mix(q[:,:,:1],k,v)[:,:,0]), 'An unrelated receiver must not alter the first receiver'
print('Rectangular scaled attention passes')''',
'postnorm_update':'''# CHECK — nonzero residual/FFN fixture distinguishes norm order
b=V1Block(8,2,16);torch.manual_seed(82)
for p in b.parameters():p.data.normal_(0,.2)
h=torch.randn(2,4,8);a=torch.randn_like(h)
z=b.norm1(h+a);expected=b.norm2(z+b.ff2(torch.nn.functional.gelu(b.ff1(z))))
assert torch.allclose(postnorm_update(h,a,b),expected), 'Two post-addition norms, with GELU only inside FFN'
print('Postnorm residual computation passes')''',
'aggregate_views':'''# CHECK — rotate back before mean logits; temperature then softmax
logits=torch.tensor([[[3.,0.,0.]],[[1.,0.,0.]]]);p=aggregate_views(logits,[0,1],3,temperature=1.)
expected=torch.softmax(torch.tensor([[1.5,0.,.5]]),-1)
assert torch.allclose(p,expected), 'Undo y->(y+s)%K in the prediction axis before averaging'
wrong=(logits[0].softmax(-1)+torch.roll(logits[1],-1,-1).softmax(-1))/2
assert not torch.allclose(p,wrong), 'Averaging probabilities is a distinct ensemble'
print('Class alignment, logit averaging and temperature pass')'''}

def check(namespace=None,save=True):
    if namespace is None:
        from relkit import tabpfn_l062_v2 as core
        namespace=vars(core)
    s=namespace;torch.set_num_threads(1);torch.manual_seed(62)
    loader,predict,normalize,outliers=official_source();path=s['ensure_checkpoint'](ROOT)
    local,_=s['load_pretrained'](path);official,_=loader(str(path.parent),path.name,'cpu');official=official[2]
    result={'wheel_sha256':WHEEL_SHA,'checkpoint_sha256':s['CHECKPOINT_SHA'],'parameter_count':sum(p.numel() for p in local.parameters())}
    x=torch.randn(1,11,100);y=torch.tensor([[0.,1.,2.,0.,1.,2.]])
    with torch.no_grad():
        a=local(x,y);b=official((x.transpose(0,1),y.T),single_eval_pos=6).transpose(0,1)
    result['pretrained_float32_max_logit_delta']=float((a-b).abs().max());assert torch.allclose(a,b,atol=1e-4,rtol=2e-5)
    local.double();official.double();xx=x.double().requires_grad_();xo=x.double().requires_grad_()
    a=local(xx,y.double());b=official((xo.transpose(0,1),y.T.double()),single_eval_pos=6).transpose(0,1)
    result['pretrained_float64_max_logit_delta']=float((a-b).detach().abs().max());assert torch.allclose(a,b,atol=1e-10,rtol=1e-10)
    a.square().mean().backward();b.square().mean().backward()
    result['input_gradient_max_delta']=float((xx.grad-xo.grad).abs().max());assert torch.allclose(xx.grad,xo.grad,atol=1e-9,rtol=1e-8)
    # Map all trained parameter gradients, not only one attention layer.
    deltas=[]
    for name,param in local.named_parameters():
        key=name.replace('blocks.','transformer_encoder.layers.').replace('qkv.weight','self_attn.in_proj_weight').replace('qkv.bias','self_attn.in_proj_bias').replace('.out.','.self_attn.out_proj.').replace('.ff1.','.linear1.').replace('.ff2.','.linear2.')
        if key.startswith('x_encoder.'):key='encoder.'+key[len('x_encoder.'):]
        if key.startswith('head.'):key='decoder.'+key[len('head.'):]
        other=dict(official.named_parameters())[key]
        assert torch.allclose(param.grad,other.grad,atol=1e-8,rtol=1e-7),key
        deltas.append(float((param.grad-other.grad).abs().max()))
    result['parameter_gradient_tensors']=len(deltas);result['parameter_gradient_max_delta']=max(deltas)
    local.float();official.float();local.zero_grad();official.zero_grad()
    cases=[]
    for kind,features,classes,views in [('binary',4,2,1),('multiclass',7,3,4),('constant',5,2,4),('missing',6,2,4),('outlier',4,2,4)]:
        torch.manual_seed(10+features);x=torch.randn(35,features);y=torch.arange(24)%classes
        if kind=='constant':x[:,1]=3
        if kind=='missing':x[3,2]=float('nan');x[27,4]=float('nan')
        if kind=='outlier':x[-1,1]=100000
        a,trace=s['predict_numeric'](local,x[:24],y,x[24:],views=views,seed=3,query_batch=5)
        b=predict(official,x[:,None],torch.cat([y,torch.zeros(11)])[:,None],24,inference_mode=True,preprocess_transform='mix',N_ensemble_configurations=views,feature_shift_decoder=True,seed=3)[0]
        delta=float((a-b).abs().max());assert torch.allclose(a,b,atol=3e-5,rtol=3e-5),(kind,delta)
        alone,_=s['predict_numeric'](local,x[:24],y,x[24:25],views=views,seed=3)
        isolation=float((alone-a[:1]).abs().max());assert isolation<3e-5
        cases.append(dict(case=kind,features=features,classes=classes,views=views,probability_max_delta=delta,query_batch_max_delta=isolation))
    result['whole_wrapper_cases']=cases
    # Context permutation vs input coordinate permutation are different symmetries.
    x=torch.randn(1,13,100);y=(torch.arange(8)%2)[None].float();order=torch.randperm(8)
    with torch.no_grad():
        a=local(x,y);p=local(torch.cat([x[:,:8][:,order],x[:,8:]],1),y[:,order])
        changed=local(x,1-y)
    assert torch.allclose(a,p,atol=1e-4,rtol=1e-5)
    result['context_permutation_max_delta']=float((a-p).abs().max());result['label_flip_max_delta']=float((a-changed).abs().max());assert result['label_flip_max_delta']>.01
    result['status']='PASS'
    if save:(ROOT/'_check_l062_v2_results.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
if __name__=='__main__':print(json.dumps(check(),indent=2))
