"""Independent numerical contracts for the L081 routing and recurrent graph model."""
import json
from pathlib import Path
import numpy as np
import torch
from relkit.mpnn_l081 import aggregate, mean_step, graph_sum, SparseGGNN

def verify():
    torch.manual_seed(81);torch.set_num_threads(1)
    h=torch.tensor([[2.],[4.],[8.],[10.]])
    edge=torch.tensor([[0,1,1,2],[1,0,2,1]])
    got=mean_step(h,edge)
    torch.testing.assert_close(got,torch.tensor([[3.],[4.5],[6.],[5.]]))
    # Repeat an edge: multigraph multiplicity is part of the input contract.
    torch.testing.assert_close(aggregate(torch.tensor([[2.],[8.],[8.]]),torch.tensor([1,1,1]),4,'mean')[1],torch.tensor([6.]))
    torch.testing.assert_close(mean_step(h,torch.empty((2,0),dtype=torch.long)),h/2)
    perm=torch.tensor([2,0,3,1]);inv=torch.argsort(perm)
    torch.testing.assert_close(mean_step(h[perm],inv[edge]),got[perm])
    torch.testing.assert_close(mean_step(h,edge[:,torch.tensor([3,1,0,2])]),got)
    x=torch.randn(4,13);etype=torch.tensor([0,0,2,2]);batch=torch.tensor([0,0,0,1])
    model=SparseGGNN(width=16,steps=3,readout_width=24)
    y=model(x,edge,etype,batch)
    torch.testing.assert_close(model(x[perm],inv[edge],etype,batch[perm]),y,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(model(x[:3],edge,etype,batch[:3]),y[:1],atol=1e-6,rtol=1e-5)
    # Rebuild bond messages with independent dense adjacency/matrix contraction.
    state=torch.randn(4,16)
    dense=torch.zeros(4,4,4)
    for k in range(edge.shape[1]):dense[etype[k],edge[1,k],edge[0,k]]+=1
    incoming=torch.einsum('kvw,kij,wj->vi',dense,model.bond_in,state)
    outgoing=torch.einsum('kvw,kij,wj->vi',dense,model.bond_out,state)
    torch.testing.assert_close(model.message(state,edge,etype),torch.cat([incoming,outgoing],1))
    # Source GRU equations checked independently in NumPy, including reset-before-matmul.
    msg=torch.randn(4,32);a=state.numpy();m=msg.numpy()
    w={k:v.detach().numpy() for k,v in model.update.named_parameters()}
    sig=lambda a:1/(1+np.exp(-a))
    z=sig(m@w['w_z']+a@w['u_z']);r=sig(m@w['w_r']+a@w['u_r'])
    expected=(1-z)*a+z*np.tanh(m@w['w']+(r*a)@w['u'])
    np.testing.assert_allclose(model.update(state,msg).detach().numpy(),expected,atol=3e-7)
    y.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    result={'status':'PASS','checks':['hand trace','duplicate edge','empty edges','node relabeling','edge order','graph isolation','dense bond contraction','NumPy source GRU equations','finite backward'], 'trace':got.flatten().tolist(),'torch':torch.__version__,'paper_result':'NOT_RUN','source_runtime_parity':'NOT_CHECKED'}
    Path(__file__).with_name('_verify_l081_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':verify()
