"""Mathematical and released-source checks; no TensorFlow runtime claim.

Execute the actual pinned TFRS Cross.call method with torch-backed Dense adapters.
This checks released call wiring under identity preactivation / zero diag_scale;
it does not check TensorFlow kernels, initialization, optimizers, or score parity.
"""
import ast
import hashlib
import json
import types
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from relkit.dcnv2 import cross_step, lowrank_step, mix_step, DCNv2


def run():
    torch.manual_seed(48)
    report = {}
    def check(name, value):
        assert value, name
        report[name] = bool(value)
    x0 = torch.randn(4,3,dtype=torch.float64,requires_grad=True)
    x = torch.randn(4,3,dtype=torch.float64,requires_grad=True)
    w = torch.randn(3,3,dtype=torch.float64,requires_grad=True)
    b = torch.randn(3,dtype=torch.float64,requires_grad=True)
    output = cross_step(x0,x,w,b)
    manual = x.detach().numpy().copy()
    for row in range(4):
        for i in range(3):
            manual[row,i] += float(x0[row,i].detach())*(sum(float(w[i,j].detach()*x[row,j].detach()) for j in range(3))+float(b[i].detach()))
    error = float(np.max(np.abs(output.detach().numpy()-manual)))
    check('Independent coordinate sums',error < 1e-12)
    check('Gradcheck input weight bias',torch.autograd.gradcheck(cross_step,(x0,x,w,b)))
    check('Bias inside product',torch.allclose(cross_step(x0,x,w*0,b),x+x0*b))
    check('Zero original coordinate preserves residual',torch.equal(cross_step(x0*0,x,w,b),x))
    u,v = [torch.randn(3,2,dtype=torch.float64,requires_grad=True) for _ in range(2)]
    check('Factorization equals dense matrix',torch.allclose(lowrank_step(x0,x,u,v,b),cross_step(x0,x,u@v.T,b)))
    check('Low rank gradcheck',torch.autograd.gradcheck(lowrank_step,(x0,x,u,v,b)))
    us,s,vh = torch.linalg.svd(w)
    exact = (us*s)@vh
    check('Full SVD reconstruction',torch.allclose(exact,w))
    # Check the literal released method, not a retyped approximation of it.
    source = Path(__file__).parent/'relkit/_references/tfrs_dcn_v073.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Cross')
    call=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='call')
    module=ast.Module(body=[call],type_ignores=[])
    env={'Optional':Optional,'tf':types.SimpleNamespace(Tensor=torch.Tensor,cast=lambda a,d:a.to(d))}
    exec(compile(module,str(source),'exec'),env)
    ref=types.SimpleNamespace(built=True,_projection_dim=None,_diag_scale=0.,compute_dtype=torch.float64,
                              _dense=lambda a:torch.nn.functional.linear(a,w,b))
    released=env['call'](ref,x0,x)
    check('Released dense call forward',torch.equal(released,output))
    grad1=torch.autograd.grad(released.square().sum(),(x0,x,w,b),retain_graph=True)
    grad2=torch.autograd.grad(output.square().sum(),(x0,x,w,b),retain_graph=True)
    check('Released dense call gradients',all(torch.equal(a,c) for a,c in zip(grad1,grad2)))
    ref._projection_dim=2
    ref._dense_u=lambda a:a@v
    ref._dense_v=lambda a:a@u.T+b
    check('Released factored call forward',torch.equal(env['call'](ref,x0,x),lowrank_step(x0,x,u,v,b)))
    # Anchor and polynomial degree: scalar weights 1, bias 0 -> x(1+x)^L.
    z=torch.linspace(-.8,.8,20,dtype=torch.float64)[:,None]
    current=z
    for depth in range(1,5):
        current=cross_step(z,current,torch.ones(1,1,dtype=z.dtype),torch.zeros(1,dtype=z.dtype))
        check(f'Anchored polynomial depth {depth}',torch.allclose(current,z*(1+z)**depth))
    # Experts: softmax gates sum to 1, residual is added once.
    ku,kv=[torch.randn(2,3,2,dtype=torch.float64,requires_grad=True) for _ in range(2)]
    kc=torch.randn(2,2,2,dtype=torch.float64,requires_grad=True)
    gates=torch.randn(4,2,dtype=torch.float64,requires_grad=True)
    pieces=[]
    for k in range(2):
        h=torch.tanh(torch.tanh(x@kv[k])@kc[k].T)
        pieces.append(x0*(h@ku[k].T+b))
    expected=x+sum(gates.softmax(-1)[:,k,None]*pieces[k] for k in range(2))
    check('Mixture equals explicit expert loop',torch.allclose(mix_step(x0,x,ku,kv,kc,b,gates),expected))
    check('Mixture gradients',torch.autograd.gradcheck(mix_step,(x0,x,ku,kv,kc,b,gates)))
    check('Zero experts add residual once',torch.equal(mix_step(x0,x,ku*0,kv,kc,b*0,gates),x))
    for kind in ('dense','lowrank','mix'):
        for layout in ('stacked','parallel'):
            model=DCNv2(3,[4],kind=kind,layout=layout).double().eval()
            a=torch.randn(5,3,dtype=torch.float64); c=torch.tensor([[0],[1],[2],[3],[0]])
            values=model(a,c)
            check(f'{kind}/{layout} output shape',values.shape==(5,))
            check(f'{kind}/{layout} independent rows',torch.allclose(values[:1],model(a[:1],c[:1]),atol=1e-12,rtol=1e-12))
            values.sum().backward()
            check(f'{kind}/{layout} cross receives gradients',all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.crosses.parameters()))
    return {'checks':report,'n_passed':len(report),'coordinate_max_error':error,
            'reference_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'reference_scope':'Pinned TFRS 0.7.3 Cross.call executed with torch-backed Dense adapters; no TensorFlow runtime or training parity claim.'}


if __name__=='__main__':
    result=run()
    Path(__file__).with_name('_check_l048_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
