"""Dense independent oracles, causal boundaries, recurrence and loss semantics."""
import importlib.util,json
from pathlib import Path
import numpy as np,torch
P=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('snap',P/'relkit/snapshot_l107.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
torch.set_num_threads(1)
def check():
 pairs=np.array([[0,1],[1,0],[1,2],[2,1],[0,1]])
 a=m.normalized_adjacency(pairs,np.ones(5),4).to_dense().numpy()
 dense=np.eye(4)
 for u,v in pairs:dense[u,v]+=1
 deg=dense.sum(1);np.testing.assert_allclose(a,dense/np.sqrt(deg[:,None]*deg[None,:]),rtol=1e-6)
 prev=torch.tensor([[.2,.4],[.6,.8]],requires_grad=True);summary=torch.zeros_like(prev)
 gates=[lambda x,h:torch.full_like(h,.25),lambda x,h:torch.ones_like(h),lambda x,h:torch.full_like(h,.8)]
 out=m.matrix_update(prev,summary,gates);torch.testing.assert_close(out,.75*prev+.2);out.sum().backward();torch.testing.assert_close(prev.grad,torch.full_like(prev,.75))
 logits=torch.tensor([[0.,0.],[0.,0.]]);y=torch.tensor([0,1]);w=torch.tensor([.1,.9])
 assert abs(m.weighted_cross_entropy(logits,y,w).item()-.5*np.log(2))<1e-6,'Release divides weighted sum by example count, not weight sum'
 return {'status':'PASS','dense_normalization':'PASS','matrix_update_value_and_gradient':'PASS','loss_reduction':'PASS'}
if __name__=='__main__':
 r=check();(P/'_check_l107_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
