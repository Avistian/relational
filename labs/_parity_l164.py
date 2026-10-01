"""Independent source parity; does not compare a port with itself."""
import copy,sys,torch
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'sources/l164/upstream'))
import relkit.griffin_l164 as port
assert hasattr(port,'GriffinMod'),'Missing checkpoint-compatible visible model'
from hmodel import GriffinMod as Released

def parity():
    torch.set_num_threads(1);torch.manual_seed(164)
    ref=Released(hiddim=16,num_mp=4,use_rev=True,use_gate=False).double().eval()
    ours=port.GriffinMod(hiddim=16,num_mp=4,use_rev=True,use_gate=False).double().eval()
    ours.load_state_dict(ref.state_dict(),strict=True)
    node=[(torch.randn(3,16,dtype=torch.float64),torch.randn(4,3,16,dtype=torch.float64)),(torch.randn(2,16,dtype=torch.float64),torch.randn(3,2,16,dtype=torch.float64))]
    mask=[torch.tensor([[0,0,1],[0,1,0],[0,0,0],[1,0,0]],dtype=torch.bool),None]
    task=[torch.randn(16,dtype=torch.float64),torch.randn(16,dtype=torch.float64)]
    edge=torch.tensor([[0,0,1,2,3,4],[4,5,6,0,2,5]])
    relation=torch.tensor([0,0,1,1,0,1]);emb=torch.randn(2,16,dtype=torch.float64)
    args=[node,mask,task,edge,relation,emb]
    a=ref(*copy.deepcopy(args));b=ours(*copy.deepcopy(args))
    torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
    w=torch.randn_like(a);(a*w).sum().backward();(b*w).sum().backward()
    err=0.
    for (n,p),(n2,q) in zip(ref.named_parameters(),ours.named_parameters()):
        assert n==n2
        if p.grad is None:assert q.grad is None
        else:
            torch.testing.assert_close(p.grad,q.grad,atol=1e-9,rtol=1e-9);err=max(err,(p.grad-q.grad).abs().max().item())
    return dict(output_max_abs=(a-b).abs().max().item(),gradient_max_abs=err,parameter_tensors=len(ref.state_dict()),mode='float64 eval; four layers; reverse edges; mixed types; masked cells')

if __name__=='__main__':print(parity())
