"""Instrumentation must preserve the original full update including BN buffers."""
import copy,json
from pathlib import Path
import torch
from torch.nn import functional as F
from relkit.debug_l116 import GCN,normalized_adjacency,train_step
P=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.manual_seed(37)
edge=torch.tensor([[0,1,2,4,5],[1,2,0,5,3]]);adj=normalized_adjacency(edge,7);x=torch.randn(7,4);y=torch.tensor([0,1,2,0,1,2,0]);idx=torch.tensor([0,3,5])
a=GCN(4,8,3,.5);b=copy.deepcopy(a);oa=torch.optim.Adam(a.parameters(),lr=.01);ob=torch.optim.Adam(b.parameters(),lr=.01)
for k in range(3):
 torch.manual_seed(100+k);loss=train_step(a,oa,x,adj,y,idx)
 torch.manual_seed(100+k);b.train();ob.zero_grad();ref=F.nll_loss(b(x,adj)[idx],y[idx]);ref.backward();ob.step()
 torch.testing.assert_close(loss,ref.detach(),rtol=0,atol=0)
 for name,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)
 for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
 a.eval();b.eval()
r={'status':'PASS','consecutive_updates':3,'optimizer':'Adam .01','dropout_rng':'MATCHED','loss_weights_gradients_bn_buffers':'EXACT','scope':'Current-runtime repaired update versus inline released-order training step'}
(P/'_trainer_check_l116_results.json').write_text(json.dumps(r,indent=2));print(r)
