"""Independent gradient/representation arithmetic and real NeighborLoader contract."""
import json,math
from pathlib import Path
import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from relkit.debug_l116 import seed_loss
P=Path(__file__).resolve().parent
r=json.loads((P/'_teaching_l116_results.json').read_text())
for row in r['gradient_chain']:assert row['input_gradient']==row['gain']**row['depth']
# Independent list arithmetic, no use of canonical sparse operator.
a=[[1,1,0],[1,1,1],[0,1,1]];d=[2,3,2];h=[2.,4.,8.];rows={x['depth']:x for x in r['smoothing']}
for k in range(65):
 if k in rows:
  assert max(abs(v-w) for v,w in zip(h,rows[k]['h']))<1e-12
  z=[v/math.sqrt(deg) for v,deg in zip(h,d)];mean=sum(z)/3;var=sum((v-mean)**2 for v in z)/3
  assert abs(var-rows[k]['degree_corrected_variance'])<1e-12
 h=[sum(a[i][j]*h[j]/math.sqrt(d[i]*d[j]) for j in range(3)) for i in range(3)]
# Real sampler confirms the hand-worked local/global convention, with all neighbors.
torch.manual_seed(7)
edge=torch.tensor([[0,2,3,5,0,5],[4,4,1,1,1,4]])
data=Data(x=torch.arange(12).float().reshape(6,2),edge_index=edge,y=torch.tensor([0,1,1,0,0,1]))
loader=NeighborLoader(data,input_nodes=torch.tensor([4,1]),num_neighbors=[-1],batch_size=2,shuffle=False)
batch=next(iter(loader));assert batch.n_id[:batch.batch_size].tolist()==[4,1]
logits=torch.randn(batch.num_nodes,2,requires_grad=True);p=logits.log_softmax(1)
loss=seed_loss(p,data.y,batch.n_id,batch.batch_size)
expected=F.nll_loss(p[:2],batch.y[:2]);torch.testing.assert_close(loss,expected)
g=torch.autograd.grad(loss,logits)[0];assert torch.count_nonzero(g[2:])==0
out={'status':'PASS','real_neighbor_loader':{'n_id':batch.n_id.tolist(),'batch_size':batch.batch_size,'seed_labels':batch.y[:2].tolist(),'context_output_supervision':'ZERO'},'independent_smoothing_steps':65,'gradient_chain_values':len(r['gradient_chain'])}
(P/'_diagnostics_check_l116_results.json').write_text(json.dumps(out,indent=2));print(out)
