"""Independent graph invariants and copied-weight parity with the pinned release."""
import importlib.util,json,ast
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from relkit.carte_l074 import make_graph,grouped_attention,Encoder
ROOT=Path(__file__).resolve().parent

def run():
    torch.set_num_threads(1)
    vec={'a':np.array([1.,2.]),'b':np.array([3.,1.]),'red':np.array([2.,-1.])}
    g=make_graph({'a':'red','b':2.,'missing':None},vec)
    assert torch.allclose(g[0][0],torch.tensor([10.,0.])), 'Center must average edge-conditioned leaves'
    assert g[0].shape==(3,2) and g[1].shape==(2,4)
    assert torch.equal(g[1],torch.tensor([[0,0,1,2],[1,2,1,2]]))
    try:make_graph({'a':None},vec)
    except ValueError:pass
    else:raise AssertionError('All-missing row needs explicit policy')
    q=torch.tensor([[1.,0.],[0.,1.]])
    ei=torch.tensor([[0,0,1],[0,1,0]])
    k=torch.tensor([[1.,0.],[0.,1.],[3.,2.]])
    v=torch.tensor([[2.,0.],[0.,4.],[7.,8.]])
    out,a=grouped_attention(ei,q,k,v)
    expected=torch.softmax(torch.tensor([1.,0.])/np.sqrt(2),0)
    assert torch.allclose(a[:2],expected) and a[2]==1
    assert torch.allclose(out[0],torch.tensor([2*expected[0],4*expected[1]]))
    assert torch.equal(out[1],v[2])
    spec=importlib.util.spec_from_file_location('upstream',ROOT/'sources/carte-l074/carte_model.py')
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    torch.manual_seed(74)
    ref=upstream.CARTE_Base(300,300,300,0,ff_dim=300,num_heads=12,dropout=0).eval()
    own=Encoder().eval();own.load_state_dict(ref.state_dict(),strict=True)
    x=torch.randn(7,300);edge=torch.tensor([[0,0,0,1,2,3,4,4,5,6],[1,2,3,1,2,3,5,6,5,6]])
    e=torch.randn(edge.shape[1],300)
    with torch.no_grad():err=float((own(x,edge,e)-ref(x,edge,e)).abs().max())
    assert err<2e-6,err
    perm=torch.tensor([3,1,6,0,5,2,4]);inverse=torch.argsort(perm)
    with torch.no_grad():assert torch.allclose(own(x[perm],inverse[edge],e)[inverse],own(x,edge,e),atol=2e-6)
    # Source graph construction, using a lightweight Data carrier and supplied vectors.
    import pandas as pd
    tree=ast.parse((ROOT/'sources/carte-l074/carte_table_to_graph.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_create_edge_index']
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Table2GraphTransformer')
    nodes.append(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_graph_construct'))
    env={'torch':torch,'np':np,'Data':lambda **kwargs:SimpleNamespace(**kwargs)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-converter>','exec'),env)
    source=env['_graph_construct'](SimpleNamespace(n_components=2),pd.Series({'a':'red'}),pd.Series({'b':2.}),np.stack(list(vec.values())),{k:i for i,k in enumerate(vec)},None,0)
    assert torch.equal(source.x,g[0]) and torch.equal(source.edge_index,g[1]) and torch.equal(source.edge_attr,g[2])
    with torch.no_grad():
        heads=torch.tensor([0,4])
        assert torch.allclose(own.rows(x,edge,e,heads),own(x,edge,e)[heads],atol=2e-6)
    weights=torch.randn(2,300)
    a=torch.autograd.grad((own.rows(x,edge,e,heads)*weights).sum(),tuple(own.parameters()))
    b=torch.autograd.grad((own(x,edge,e)[heads]*weights).sum(),tuple(own.parameters()))
    grad_error=max(float((u-v).abs().max()) for u,v in zip(a,b))
    assert all(torch.allclose(u,v,atol=5e-5,rtol=5e-5) for u,v in zip(a,b)),grad_error
    # Test the actual selected checkpoint as well as a random copied state.
    from relkit.carte_l074 import load_encoder
    own=load_encoder(ROOT/'data/l074/kg_pretrained.pt',True,74).eval()
    ref.load_state_dict(own.state_dict(),strict=True)
    with torch.no_grad():pre_error=float((own(x,edge,e)-ref(x,edge,e)).abs().max())
    assert pre_error<2e-6,pre_error
    report={'status':'PASS','center_only_gradient_max_error':grad_error,'pretrained_output_error':pre_error,'source_graph_parity':True,'max_output_error':err,'grouped_attention':True,'permutation_equivariance':True,'missing_row_rejection':True}
    (ROOT/'_check_l074_results.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':run()
