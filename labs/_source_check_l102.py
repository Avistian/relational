"""Independent released-source forward, state and gradient comparison (copied weights)."""
import json,sys
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'sources/l102'))
from model.tgn import TGN as Released
from utils.utils import get_neighbor_finder
from utils.data_processing import Data
from relkit.tgn_l102 import TGN,temporal_neighbors

def main(device="cpu"):
 torch.set_num_threads(1)
 rng=np.random.RandomState(11)
 nodes=rng.normal(size=(9,4)).astype('float32');nodes[0]=0
 edges=rng.normal(size=(13,4)).astype('float32');edges[0]=0
 u=np.array([1,2,1,3,2,4,1,4,2,3,1,4]);v=np.array([5,6,7,5,8,6,8,7,5,8,6,5]);t=np.arange(1,13,dtype=float);ei=np.arange(1,13)
 data=Data(u,v,t,ei,np.zeros(12))
 finder=get_neighbor_finder(data,False)
 torch.manual_seed(9)
 release=Released(finder,nodes,edges,device,n_layers=1,n_heads=2,dropout=0,use_memory=True,
                  memory_dimension=4,message_function='identity',aggregator_type='last',memory_updater_type='gru').to(device)
 torch.manual_seed(9);port=TGN(nodes,edges,dropout=0,neighbors=3).to(device)
 port.finder=temporal_neighbors(u,v,t,ei,len(nodes))
 mapping={'time_encoder':'time_encoder','gru':'memory_updater.memory_updater',
          'merge':'embedding_module.attention_models.0.merger',
          'attention':'embedding_module.attention_models.0.multi_head_target','decoder':'affinity_score'}
 rp=dict(release.named_parameters()); pp=dict(port.named_parameters())
 initial=0
 for name,p in pp.items():
  prefix,tail=name.split('.',1);ref=mapping[prefix]+'.'+tail
  initial=max(initial,float((p-rp[ref]).abs().max().detach()))
  p.data.copy_(rp[ref].data)
 maxout=maxgrad=maxstate=0.
 for lo in range(0,12,3):
  s=slice(lo,lo+3);neg=np.array([8,7,6]);release.zero_grad();port.zero_grad()
  a=release.compute_edge_probabilities(u[s],v[s],neg,t[s],ei[s],3)
  b=port.probabilities(u[s],v[s],neg,t[s],ei[s])
  for x,y in zip(a,b):
   maxout=max(maxout,float((x.flatten()-y).abs().max().detach()))
   assert torch.allclose(x.flatten(),y,atol=2e-6,rtol=2e-5)
  la=sum(x.sum() for x in a);lb=sum(x.sum() for x in b);la.backward();lb.backward()
  for name,p in pp.items():
   prefix,tail=name.split('.',1);q=rp[mapping[prefix]+'.'+tail]
   assert (p.grad is None)==(q.grad is None),name
   if p.grad is not None:
    maxgrad=max(maxgrad,float((p.grad-q.grad).abs().max()))
    assert torch.allclose(p.grad,q.grad,atol=3e-6,rtol=3e-4),name
  maxstate=max(maxstate,float((port.memory-release.memory.memory).abs().max().detach()))
  assert torch.allclose(port.memory,release.memory.memory,atol=2e-6)
  for node,(message,when) in port.pending.items():
   assert torch.allclose(message,release.memory.messages[node][-1][0],atol=2e-6)
  release.memory.detach_memory();port.detach_state()
 result={'status':'PASS','batches':4,'events':12,'initial_parameter_max_error':initial,
         'max_probability_error':maxout,'max_gradient_error':maxgrad,'max_state_error':maxstate,
         'device':device,'scope':'One-layer TGN-attn repeated-node stream, empty history, raw pending messages; dropout disabled',
         'source_commit':'e38cdf85998c6ca077167610dc4e769a688efa95'}
 (P/'_source_check_l102_results.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'cpu')
