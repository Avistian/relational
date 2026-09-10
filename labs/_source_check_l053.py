"""Execute unmodified pinned author primitives; copy weights for full forward/gradient parity."""
import ast,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator,TransformerMixin
import sklearn
from relkit.realmlp import RealMLPS,RobustSmooth,coslog4

ROOT=Path(__file__).resolve().parent
def source_classes(file,names,env):
    tree=ast.parse(file.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),env)

def check():
    manifest=json.loads((ROOT/'_sources_l053.json').read_text());src=ROOT/'sources/l053'
    for name,info in manifest['files'].items():assert hashlib.sha256((src/name).read_bytes()).hexdigest()==info['sha256']
    env=dict(torch=torch,nn=nn,np=np,sklearn=sklearn)
    source_classes(src/'mlp.py',{'ScalingLayer','NTPLinear','Mish'},env)
    source_classes(src/'preprocessing.py',{'RobustScaleSmoothClipTransform'},env)
    x=np.random.default_rng(53).normal(size=(100,7));x[:,2]=1;x[:99,3]=0;x[-1,3]=100
    q=np.random.default_rng(54).normal(size=(27,7))*100
    ref=env['RobustScaleSmoothClipTransform']().fit(x).transform(q)
    ours=RobustSmooth().fit(x).transform(q)
    np.testing.assert_allclose(ours,ref,rtol=1e-6,atol=1e-6)
    cases=[]
    # Execute the author's optimizer construction and group-rate assignments,
    # rather than constructing a second optimizer from our own recipe constants.
    author_tree=ast.parse((src/'mlp.py').read_text())
    fit=next(n for n in ast.walk(author_tree) if isinstance(n,ast.FunctionDef) and n.name=='fit')
    opt_start=next(i for i,n in enumerate(fit.body) if isinstance(n,ast.Assign) and
                   any(isinstance(t,ast.Name) and t.id=='params' for t in n.targets))
    opt_end=next(i for i,n in enumerate(fit.body) if isinstance(n,ast.Assign) and
                 any(isinstance(t,ast.Name) and t.id=='opt' for t in n.targets))
    optimizer_code=compile(ast.Module(body=fit.body[opt_start:opt_end+1],type_ignores=[]),'<author optimizer>','exec')
    rate_assignments=[n for n in ast.walk(fit) if isinstance(n,ast.Assign) and
                      any('opt.param_groups' in ast.unparse(t) for t in n.targets)]
    assert len(rate_assignments)==3
    rate_code=compile(ast.Module(body=rate_assignments,type_ignores=[]),'<author rates>','exec')
    for regression in [False,True]:
        torch.manual_seed(53);model=RealMLPS(7,width=16,regression=regression)
        # Nonzero output weights exercise every hidden layer and its input gradients.
        with torch.no_grad():model.layers[-1].weight.normal_();model.layers[-1].bias.normal_();model.scale.uniform_(.2,1.5)
        act=env['Mish'] if regression else nn.SELU
        reference=nn.Sequential(env['ScalingLayer'](7),env['NTPLinear'](7,16),act(),
            env['NTPLinear'](16,16),act(),env['NTPLinear'](16,16),act(),env['NTPLinear'](16,1 if regression else 2))
        with torch.no_grad():
            reference[0].scale.copy_(model.scale)
            for a,b in zip(model.layers,[reference[i] for i in [1,3,5,7]]):b.weight.copy_(a.weight);b.bias.copy_(a.bias)
        a=torch.randn(11,7,requires_grad=True);b=a.detach().clone().requires_grad_()
        p=model(a);r=reference(b);torch.testing.assert_close(p,r,rtol=2e-5,atol=2e-6)
        p.square().mean().backward();r.square().mean().backward()
        torch.testing.assert_close(a.grad,b.grad,rtol=2e-5,atol=2e-6)
        parameter_error=0.
        for left,right in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(left.grad,right.grad,rtol=3e-5,atol=3e-6)
            parameter_error=max(parameter_error,float((left.grad-right.grad).abs().max()))
        optimizer=torch.optim.Adam(model.parameter_groups(),betas=(.9,.95),eps=1e-8,weight_decay=0.)
        author_env=dict(model=reference,torch=torch)
        exec(optimizer_code,author_env);author_opt=author_env['opt']
        state_error=0.;update_error=0.
        for t in [0.,.037,.25,.8]:
            optimizer.zero_grad();author_opt.zero_grad()
            target=torch.linspace(-.5,.5,11) if regression else torch.arange(11)%2
            left=model(a.detach());right=reference(b.detach())
            if regression:
                loss=(left[:,0]-target).square().mean();reference_loss=(right[:,0]-target).square().mean()
            else:
                loss=nn.functional.cross_entropy(left,target,label_smoothing=.1)
                reference_loss=nn.functional.cross_entropy(right,target,label_smoothing=.1)
            loss.backward();reference_loss.backward()
            rate=(.07 if regression else .04)*coslog4(t)
            for group in optimizer.param_groups:group['lr']=rate*group['factor']
            exec(rate_code,dict(author_env,lr=rate))
            before=[p.detach().clone() for p in model.parameters()]
            optimizer.step();author_opt.step()
            for left,right,initial in zip(model.parameters(),reference.parameters(),before):
                torch.testing.assert_close(left,right,rtol=3e-5,atol=3e-6)
                update_error=max(update_error,float((left-right).abs().max().detach()))
                for key in ['exp_avg','exp_avg_sq','step']:
                    xstate=optimizer.state[left][key];ystate=author_opt.state[right][key]
                    torch.testing.assert_close(xstate,ystate,rtol=3e-5,atol=3e-6)
                    state_error=max(state_error,float((xstate-ystate).abs().max()))
                if t==0:torch.testing.assert_close(left,initial,rtol=0,atol=0)
        cases.append(dict(regression=regression,output_max_error=float((p-r).abs().max().detach()),gradient_max_error=float((a.grad-b.grad).abs().max()),
                          parameter_gradient_max_error=parameter_error,optimizer_parameter_max_error=update_error,optimizer_state_max_error=state_error))
    # Extract and execute the authors' exact schedule expression (inside SimpleMLP.fit).
    tree=ast.parse((src/'mlp.py').read_text())
    expression=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='lr_sched_value' for t in n.targets))
    schedule=compile(ast.Expression(expression),'<author schedule>','eval')
    for t in np.linspace(0,1,101):assert abs(coslog4(float(t))-eval(schedule,{'np':np,'t':t}))<1e-12
    result=dict(status='PASS',revision=manifest['revision'],cases=cases,preprocess_max_error=float(np.max(abs(ref-ours))),
                scope='Pinned primitive execution; numeric forward/input/parameter-gradient parity with copied nonzero weights; four matched-batch Adam updates and moment states using author optimizer/group assignments, both task types; 101 schedule points. Not full shuffled training, initialization-stream, validation-selection or pytabkit runtime parity.')
    (ROOT/'_source_check_l053_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':check()
