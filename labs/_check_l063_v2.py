"""Read-only live-function and pinned original-source checker for L063."""
import ast, copy, hashlib, json, math, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from relkit import scm_l063_v2 as core
ROOT=Path(__file__).resolve().parent
CHECKS={
'sparse_weights':'''a=np.array([[1.,2.],[3.,4.]]);m=np.array([[1,0],[0,1]])
np.testing.assert_allclose(sparse_weights(a,m,.25),[[2,0],[0,8]])
np.testing.assert_array_equal(sparse_weights(a,m,.25,first=True),a)
try:sparse_weights(a,m,1.)
except ValueError:pass
else:raise AssertionError('Dropout1 must fail')''',
'propagate':'''c=np.array([[1.,-2.]])
w=[np.array([[2.,0.],[0.,1.]]),np.array([[1.,-1.]]),np.array([[2.]])]
b=[np.array([0.,1.]),np.array([.5]),np.array([-.2])];e=[np.array([[.1]]),np.array([[.3]])]
h=propagate(c,w,b,e,'identity')
np.testing.assert_allclose(h[0],[[2.,-1.]])
np.testing.assert_allclose(h[1],[[3.6]])
np.testing.assert_allclose(h[2],[[7.3]])
# Activation-before-affine is distinguishable from activation-after-affine.
t=propagate(c,w,b,e,'tanh')
np.testing.assert_allclose(t[1],[[np.tanh(2)-np.tanh(-1)+.6]])
try:propagate(c,w,b,[np.zeros((2,1)),e[1]],'tanh')
except ValueError:pass
else:raise AssertionError('One noise per row/node required')''',
'select_nodes':'''v=np.arange(12.).reshape(3,4)
x,z=select_nodes(v,[2,0],3)
np.testing.assert_array_equal(x,v[:,[2,0]])
for f in [[0,3],[0,0],[-1,0]]:
    try:select_nodes(v,f,3)
    except ValueError:pass
    else:raise AssertionError('Reject target overlap, duplicates and invalid indices')''',
'rank_labels':'''z=np.array([-1.,-.1,.2,.5,1.])
y,b=rank_labels(z,np.array([1,3]),np.array([2,0,1]))
np.testing.assert_array_equal(y,[2,2,0,0,1])
# Repeated sampled bounds create an empty middle interval, not an error.
y,b=rank_labels(z,np.array([1,1]),np.arange(3))
assert set(y)=={0,2}
assert b[0]==b[1]''',
'posterior_weights':'''w=posterior_weights(np.log([.5,.5]),np.log([.4,.6]),np.log([.2,.8]))
np.testing.assert_allclose(w,[1/7,6/7])
assert abs(w@np.array([.1,.9])-11/14)<1e-12
np.testing.assert_allclose(posterior_weights(np.array([-10000.,-10001.]),np.zeros(2),np.zeros(2)),[.7310585786300049,.2689414213699951])
try:posterior_weights(np.array([-np.inf,-np.inf]),np.zeros(2),np.zeros(2))
except ValueError:pass
else:raise AssertionError('Impossible observation must fail')'''}

def check(namespace=None,save=False):
    live=vars(core) if namespace is None else namespace
    for name,code in CHECKS.items():exec(code,live.copy())
    manifest=json.loads((ROOT/'_sources_l063_v2.json').read_text())
    for row in manifest['sources']:
        assert hashlib.sha256((ROOT.parent/row['path']).read_bytes()).hexdigest()==row['sha256']
    src=ast.parse((ROOT/'sources/l063-v2/mlp.py').read_text())
    items=[n for n in src.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['GaussianNoise','causes_sampler_f']]
    gb=next(n for n in src.body if isinstance(n,ast.FunctionDef) and n.name=='get_batch')
    original_class=next(n for n in gb.body if isinstance(n,ast.ClassDef) and n.name=='MLP')
    env=dict(torch=torch,nn=nn,np=np,math=math,random=random,device='cpu',seq_len=23,num_features=3,num_outputs=1)
    exec(compile(ast.Module(body=items+[original_class],type_ignores=[]),'pinned-mlp.py','exec'),env)
    activation_map={'identity':nn.Identity,'tanh':nn.Tanh,'leaky_relu':nn.LeakyReLU,'elu':nn.ELU}
    errors=[];cases=[]
    for family in ['SCM','BNN']:
      for block in [False,True]:
       for activation,cls in activation_map.items():
        torch.manual_seed(63);np.random.seed(63);random.seed(63)
        hp=dict(num_layers=4,num_causes=3,prior_mlp_hidden_dim=9,noise_std=.15,init_std=.3,is_causal=family=='SCM',prior_mlp_activations=cls,pre_sample_causes=True,pre_sample_weights=True,block_wise_dropout=block,prior_mlp_dropout_prob=.36,prior_mlp_scale_weights_sqrt=True,sampling='normal',in_clique=False,y_is_effect=True,sort_features=False,random_feature_rotation=False)
        env['hyperparameters']=hp;model=env['MLP'](hp).double()
        # Original forward forces causes.float(); run original source in float32.
        model=model.float();seen={};outputs=[];noises=[]
        handles=[model.layers[0].register_forward_pre_hook(lambda module,args:seen.update(causes=args[0].detach().numpy()[:,0].copy()))]
        for layer in model.layers:
            handles.append(layer.register_forward_hook(lambda module,args,out:outputs.append(out.detach().numpy()[:,0].copy())))
            if isinstance(layer,nn.Sequential):
                handles.append(layer[-1].register_forward_hook(lambda module,args,out:noises.append((out-args[0]).detach().numpy()[:,0].copy())))
        x,y=model()
        for h in handles:h.remove()
        ws=[];bs=[]
        for layer in model.layers:
            linear=layer if isinstance(layer,nn.Linear) else layer[1]
            ws.append(linear.weight.detach().numpy().copy());bs.append(linear.bias.detach().numpy().copy())
        own=live['propagate'](seen['causes'],ws,bs,noises,activation)
        error=max(float(np.max(np.abs(a-b))) for a,b in zip(own,outputs))
        assert error<2e-5,(family,activation,block,error)
        errors.append(error);cases.append(dict(family=family,activation=activation,block_dropout=block,max_abs_error=error))
    # Validate learner's sparsifier against the exact original initializer expression under fixed draws.
    raw=np.array([[1.,2.],[3.,4.]]);mask=np.array([[1,0],[0,1]])
    for p in [0.,.25,.81]:np.testing.assert_allclose(live['sparse_weights'](raw,mask,p),raw*mask/(1-p**.5))
    # Execute the original MulticlassRank and randomize_classes definitions.
    rank_tree=ast.parse((ROOT/'sources/l063-v2/flexible_categorical.py').read_text());util=ast.parse((ROOT/'sources/l063-v2/utils.py').read_text())
    chosen=[n for n in rank_tree.body if isinstance(n,ast.ClassDef) and n.name=='MulticlassRank']+[n for n in util.body if isinstance(n,ast.FunctionDef) and n.name=='randomize_classes']
    rank_env=dict(torch=torch,nn=nn,class_sampler_f=lambda a,b:lambda:3)
    exec(compile(ast.Module(body=chosen,type_ignores=[]),'pinned-rank.py','exec'),rank_env)
    rank_error=0
    for seed in range(20):
        target=torch.tensor([-1.,-.1,.2,.5,1.]).reshape(-1,1)
        torch.manual_seed(seed);ref=rank_env['MulticlassRank'](3,ordered_p=.5)(target.clone()).numpy()[:,0]
        torch.manual_seed(seed);idx=torch.randint(0,len(target),(2,)).numpy();perm=np.arange(3)
        randomized=bool(torch.rand((1,))>.5);drawn_perm=torch.randperm(3).numpy()
        if randomized:perm=drawn_perm
        if bool(torch.rand((1,))>.5):perm=2-perm
        own,_=live['rank_labels'](target.numpy()[:,0],idx,perm)
        np.testing.assert_array_equal(own,ref)
    result=dict(status='PASS',live_tasks=list(CHECKS),upstream_forward_cases=cases,max_forward_error=max(errors),multiclass_source_cases=20,source_sha256_checked=len(manifest['sources']),scope='Pinned original MLP class executed with its copied weights, recorded causes and additive noise; float32 original versus float64 NumPy. Conditional computation parity only. Sampler RNG and complete distribution NOT_PARITY.')
    if save:(ROOT/'_check_l063_v2_results.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
if __name__=='__main__':print(json.dumps(check(save=True),indent=2))
