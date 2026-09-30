"""Independent PyG outputs/gradient checks and full neural fixtures."""
import copy,json,types
from pathlib import Path
import torch
from torch_geometric.nn import TransformerConv
from relkit.relgnn_l142 import OrdinaryGNN,RelGNN_Model,OrdinaryModel,get_atomic_routes
from _fixture_l141 import neural_fixture
from _parity_l141 import original_modules

def check():
    torch.manual_seed(142);torch.set_num_threads(1)
    edges={('fact','f2p_a','a'):torch.tensor([[0,1,2],[0,0,1]]),('a','rev_f2p_a','fact'):torch.tensor([[0,0,1],[0,1,2]]),('fact','f2p_b','b'):torch.tensor([[0,1,2],[0,1,0]]),('b','rev_f2p_b','fact'):torch.tensor([[0,1,0],[0,1,2]])}
    a=OrdinaryGNN(['fact','a','b'],list(edges),4,2).double();b=copy.deepcopy(a)
    def oracle(conv,x,edge):return conv.final_proj(TransformerConv.forward(conv,x,edge))
    for layer in b.convs:
        for conv in layer.values():conv.forward=types.MethodType(oracle,conv)
    x={n:torch.randn(size,4,dtype=torch.float64,requires_grad=True) for n,size in [('fact',3),('a',2),('b',2)]}
    y={n:v.detach().clone().requires_grad_() for n,v in x.items()}
    out=a(x,edges);ref=b(y,edges)
    for n in x:torch.testing.assert_close(out[n],ref[n],atol=1e-11,rtol=1e-11)
    sum(v.square().sum() for v in out.values()).backward();sum(v.square().sum() for v in ref.values()).backward()
    count=0
    for (n,p),(m,q) in zip(a.named_parameters(),b.named_parameters()):
        assert n==m;torch.testing.assert_close(p.grad,q.grad,atol=1e-10,rtol=1e-10);count+=1
    for n in x:torch.testing.assert_close(x[n].grad,y[n].grad,atol=1e-10,rtol=1e-10)
    def ordinary_factory(*args,**kwargs):
        kwargs.update(dict(zip(['data','col_stats_dict','num_model_layers','channels','out_channels','aggr','norm'],args)))
        return OrdinaryModel(**kwargs)
    fixtures={name:neural_fixture(model,get_atomic_routes) for name,model in [('composite',RelGNN_Model),('ordinary',ordinary_factory)]}
    # Exact inherited composite operator oracle, using L142 live aggregation/fusion.
    import _parity_l141 as parity
    src=Path(__file__).parent/'sources/l141'
    text=(Path(__file__).parent/'_parity_l141.py').read_text().split("if __name__=='__main__':")[0].replace('from relkit.relgnn_l141 import RelGNNConv','from relkit.relgnn_l142 import RelGNNConv')
    namespace={};exec(text,namespace);composite=namespace['check'](src)
    return dict(status='PASS',ordinary_gradient_tensors=count,composite=composite,fixtures=fixtures)

if __name__=='__main__':
    r=check();Path(__file__).with_name('_parity_l142_results.json').write_text(json.dumps(r,indent=2));print(r)
