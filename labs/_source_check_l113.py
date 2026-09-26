"""Compare the visible operator to the publication-era released model class."""
import ast,json,importlib.metadata
from pathlib import Path
import torch
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv
from relkit.scaling_l113 import SAGE,mean_adjacency

def check(device='cpu'):
    src=(Path(__file__).parent/'sources/l113/cluster_gcn.py').read_text()
    cls=next(n for n in ast.parse(src).body if isinstance(n,ast.ClassDef) and n.name=='SAGE')
    space={'torch':torch,'F':F,'SAGEConv':SAGEConv};exec(compile(ast.Module(body=[cls],type_ignores=[]),'pinned-release','exec'),space)
    results=[]
    for training in [False,True]:
        torch.manual_seed(113);ours=SAGE(3,7,2,3,0.).to(device);original=space['SAGE'](3,7,2,3,0.).to(device)
        original.load_state_dict(ours.state_dict());ours.train(training);original.train(training)
        edge=torch.tensor([[0,0,2,1,2,4,1],[1,1,1,2,0,2,1]],device=device);a=mean_adjacency(edge,6)
        x=torch.randn(6,3,device=device,requires_grad=True);xx=x.detach().clone().requires_grad_()
        y=ours(x,a);yy=original(xx,edge);torch.testing.assert_close(y,yy,atol=2e-6,rtol=2e-5)
        y.square().sum().backward();yy.square().sum().backward();torch.testing.assert_close(x.grad,xx.grad,atol=2e-6,rtol=2e-5)
        for name,p in ours.named_parameters():torch.testing.assert_close(p.grad,dict(original.named_parameters())[name].grad,atol=2e-6,rtol=2e-5)
        results.append({'training':training,'max_output_error':float((y-yy).abs().max().detach()),'max_input_gradient_error':float((x.grad-xx.grad).abs().max())})
    return {'status':'PASS','device':device,'runtime':{k:importlib.metadata.version(k) for k in ['torch','torch-geometric']},'cases':results,'scope':'Copied-weight forward and gradients; dropout disabled; not historical runtime or training trajectory identity'}
if __name__=='__main__':
    r=check();(Path(__file__).parent/'_source_check_l113_results.json').write_text(json.dumps(r,indent=2));print(r)
