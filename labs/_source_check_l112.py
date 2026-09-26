"""Compare loaded parameters, forward, gradients and an Adam step to original GCN."""
import importlib.util,json,sys
from pathlib import Path
import torch
from relkit.ogb_l112 import GCN,normalized_adjacency
P=Path(__file__).resolve().parent

def original():
    sys.path.insert(0,str(P/'sources/l112'))
    spec=importlib.util.spec_from_file_location('ogb_original',P/'sources/l112/gnn.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.GCN

def transfer(model,ref):
    state=model.state_dict();converted={}
    for k,v in state.items():
        converted[k.replace('.weight','.lin.weight') if k.startswith('convs.') and k.endswith('.weight') else k]=v
    ref.load_state_dict(converted)

def check(device='cpu'):
    torch.manual_seed(4);torch.set_num_threads(1)
    edge=torch.tensor([[0,1,2,3,3,5],[1,2,0,2,4,4]],device=device)
    binary=torch.cat([edge,edge.flip(0)],1);binary=torch.unique(binary,dim=1)
    adj=normalized_adjacency(edge,7).to(device)
    model=GCN(4,8,3).to(device);ref=original()(4,8,3,3,.5).to(device);transfer(model,ref)
    maxima={}
    for training in [False,True]:
        model.train(training);ref.train(training)
        x=torch.randn(7,4,device=device);x1=x.clone().requires_grad_();x2=x.clone().requires_grad_()
        torch.manual_seed(99);a=model(x1,adj)
        torch.manual_seed(99);b=ref(x2,binary)
        torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
        model.zero_grad();ref.zero_grad();a.square().mean().backward();b.square().mean().backward()
        torch.testing.assert_close(x1.grad,x2.grad,rtol=3e-4,atol=3e-6)
        for name,p in model.named_parameters():
            key=name.replace('.weight','.lin.weight') if name.startswith('convs.') and name.endswith('.weight') else name
            q=dict(ref.named_parameters())[key]
            torch.testing.assert_close(p.grad,q.grad,rtol=3e-4,atol=3e-6)
        maxima[str(training)]=float((a-b).detach().abs().max())
    # Use eval-mode BN for the Adam comparison: training-mode pre-BN bias
    # gradients are analytically zero, where float32 noise is amplified by Adam.
    model.eval();ref.eval();model.zero_grad();ref.zero_grad()
    model(x1,adj).square().mean().backward();ref(x2,binary).square().mean().backward()
    opt1=torch.optim.Adam(model.parameters(),lr=.01);opt2=torch.optim.Adam(ref.parameters(),lr=.01);opt1.step();opt2.step()
    for name,p in model.named_parameters():
        key=name.replace('.weight','.lin.weight') if name.startswith('convs.') and name.endswith('.weight') else name
        torch.testing.assert_close(p,dict(ref.named_parameters())[key],rtol=3e-4,atol=3e-6)
    return {'status':'PASS','device':device,'max_output_absolute_error':maxima,'gradients':'PASS','adam_step':'PASS','scope':'Identical loaded parameters and dropout RNG, current runtime; not historical initialization identity'}
if __name__=='__main__':
    r=check(sys.argv[1] if len(sys.argv)>1 else 'cpu');(P/'_source_check_l112_results.json').write_text(json.dumps(r,indent=2));print(r)
