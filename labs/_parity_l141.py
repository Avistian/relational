"""Original source differential check, including outputs and all active gradients."""
import contextlib,sys,types
from pathlib import Path
import torch

@contextlib.contextmanager
def original_modules(source_root):
    names=['atomic_routes','relgnn_conv','relgnn_hetero_conv','relgnn_nn','relgnn_model']
    old={n:sys.modules.get(n) for n in names}
    try:
        for name in names:
            m=types.ModuleType(name);sys.modules[name]=m
            exec(compile((Path(source_root)/('examples__'+name+'.py')).read_text(),name,'exec'),m.__dict__)
        yield sys.modules['relgnn_model'].RelGNN_Model,sys.modules['relgnn_conv'].RelGNNConv
    finally:
        for n,m in old.items():
            if m is None:sys.modules.pop(n,None)
            else:sys.modules[n]=m

def check(source_root):
    from relkit.relgnn_l141 import RelGNNConv
    torch.manual_seed(141);torch.set_num_threads(1)
    errors=[];gradients=0
    with original_modules(source_root) as (_,Original):
        for mode in ['dim-dim','dim-fact-dim']:
            for empty in [False,True]:
                a=RelGNNConv(mode,(4,4),4,2,aggr='sum').double()
                b=Original(mode,(4,4),4,2,aggr='sum').double();b.load_state_dict(a.state_dict())
                x=[torch.randn(n,4,dtype=torch.float64,requires_grad=True) for n in ([3,2] if mode=='dim-dim' else [3,4,2])]
                y=[v.detach().clone().requires_grad_() for v in x]
                e=torch.tensor([[0,1,2],[0,0,1]]) if not empty else torch.empty((2,0),dtype=torch.long)
                edges=e if mode=='dim-dim' else (e,torch.tensor([[0,1,2],[0,1,2]]))
                oa=a(tuple(x),edges);ob=b(tuple(y),edges)
                aa=(oa,) if mode=='dim-dim' else oa;bb=(ob,) if mode=='dim-dim' else ob
                for u,v in zip(aa,bb):
                    torch.testing.assert_close(u,v,atol=1e-12,rtol=1e-12);errors.append(float((u-v).abs().max().detach()))
                sum(u.square().sum() for u in aa).backward();sum(u.square().sum() for u in bb).backward()
                for (ka,pa),(kb,pb) in zip(a.named_parameters(),b.named_parameters()):
                    assert ka==kb
                    if pa.grad is not None:
                        torch.testing.assert_close(pa.grad,pb.grad,atol=1e-11,rtol=1e-11);gradients+=1
                for u,v in zip(x,y):torch.testing.assert_close(u.grad,v.grad,atol=1e-11,rtol=1e-11)
    return dict(status='PASS',cases=4,parameter_gradient_tensors=gradients,max_output_error=max(errors),dtype='float64',scope='operator outputs/input and parameter gradients; empty attention edges included')
if __name__=='__main__':
    import json
    p=Path(__file__).resolve().parent;r=check(p/'sources/l141');(p/'_parity_l141_results.json').write_text(json.dumps(r,indent=2));print(r)
