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
        cases.append(dict(regression=regression,output_max_error=float((p-r).abs().max().detach()),gradient_max_error=float((a.grad-b.grad).abs().max())))
    # Extract and execute the authors' exact schedule expression (inside SimpleMLP.fit).
    tree=ast.parse((src/'mlp.py').read_text())
    expression=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='lr_sched_value' for t in n.targets))
    schedule=compile(ast.Expression(expression),'<author schedule>','eval')
    for t in np.linspace(0,1,101):assert abs(coslog4(float(t))-eval(schedule,{'np':np,'t':t}))<1e-12
    result=dict(status='PASS',revision=manifest['revision'],cases=cases,preprocess_max_error=float(np.max(abs(ref-ours))),
                scope='Pinned primitive execution; full numeric forward/input-gradient parity with copied weights; 101 schedule points. No full optimizer trajectory or pytabkit runtime parity claimed.')
    (ROOT/'_source_check_l053_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':check()
