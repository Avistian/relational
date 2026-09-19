"""Independent numerical, invariance, eligibility and gradient checks."""
import hashlib,json,importlib.metadata
from pathlib import Path
import torch
from torch_geometric.nn import SAGEConv
from relkit.stack_l076 import (fixture,key_positions,eligible_events,mean_messages,
                               materialize_tables,RelationalStack,run_diagnostic)
ROOT=Path(__file__).resolve().parent

def run():
 torch.set_num_threads(1);torch.manual_seed(76)
 c,e,y=fixture();cd,ed,et,dest,mask=materialize_tables(c,e)
 assert dest.tolist()==[1,0,1,2,0] and mask.tolist()==[True,True,True,False,False]
 for pk,fk in [([1,1],[1]),([1],[2])]:
  try:key_positions(pk,fk)
  except ValueError:pass
  else:raise AssertionError('Broken key integrity accepted')
 # Hand oracle, duplicate destinations and isolated nodes.
 messages=torch.tensor([[2.,1.],[4.,3.],[6.,5.]],requires_grad=True)
 positions=torch.tensor([1,0,1]);actual,counts=mean_messages(messages,positions,4)
 expected=torch.tensor([[4.,3.],[4.,3.],[0.,0.],[0.,0.]])
 assert torch.equal(actual,expected) and counts.tolist()==[1,2,0,0]
 dense=torch.zeros(4,3);dense[positions,torch.arange(3)]=1
 oracle=dense@messages/dense.sum(1).clamp_min(1)[:,None]
 assert torch.equal(actual,oracle)
 grad_a=torch.autograd.grad(actual.square().sum(),messages,retain_graph=True)[0]
 grad_b=torch.autograd.grad(oracle.square().sum(),messages)[0]
 assert torch.equal(grad_a,grad_b)
 # Reference primitive: SAGEConv mean with identity neighbor transform and no root.
 sage=SAGEConv((2,2),2,aggr='mean',root_weight=False,bias=False)
 with torch.no_grad():sage.lin_l.weight.copy_(torch.eye(2))
 ref=sage((messages,torch.zeros(4,2)),torch.stack([torch.arange(3),positions]))
 assert torch.allclose(ref,actual,atol=1e-7)
 empty,n=mean_messages(messages[:0],positions[:0],4)
 assert torch.equal(empty,torch.zeros(4,2)) and not n.any()
 model=RelationalStack(cd,ed);logits,tr=model(cd.tensor_frame,et,dest,mask)
 assert logits.shape==(4,) and tr['combined'].shape==(4,8)
 loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y);loss.backward()
 for block in [model.customer_encoder,model.event_encoder,model.head]:
  grads=[p.grad for p in block.parameters() if p.grad is not None]
  assert grads and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads)>0
 # Future values cannot affect fitted state OR outputs (not just edges).
 changed=e.copy();changed.loc[~mask.numpy(),'amount']=1e8
 cd2,ed2,et2,d2,m2=materialize_tables(c,changed)
 assert cd.col_stats==cd2.col_stats and ed.col_stats==ed2.col_stats
 assert torch.equal(logits,model(cd2.tensor_frame,et2,d2,m2)[0])
 # Reorder event rows; reductions remain unchanged to numerical tolerance.
 order=torch.tensor([4,2,0,3,1]);p=model(cd.tensor_frame,et[order],dest[order],mask[order])[0]
 assert torch.allclose(logits,p,atol=1e-7)
 # Reorder customers AND remap FK; outputs follow identities, not integer positions.
 order=torch.tensor([2,0,3,1]);d3=key_positions(c.customer_id.iloc[order.numpy()].tolist(),e.customer_id.tolist())
 p=model(cd.tensor_frame[order],et,d3,mask)[0]
 assert torch.allclose(logits[order],p,atol=1e-7)
 # Event-row gradient support: no path from excluded event embeddings to target.
 model.zero_grad();out,trace=model(cd.tensor_frame,et,dest,mask);trace['events'].retain_grad();out.sum().backward()
 g=trace['events'].grad
 assert torch.equal(g[~mask],torch.zeros_like(g[~mask])) and g[mask].abs().sum()>0
 diagnostic=run_diagnostic();assert diagnostic['final_loss']<diagnostic['first_loss']/10
 result={'status':'PASS','implementation_sha256':hashlib.sha256((ROOT/'relkit/stack_l076.py').read_bytes()).hexdigest(),
         'checks':['PK/FK integrity','dense forward/backward oracle','PyG mean primitive parity','empty neighborhoods','end-to-end gradients','future fit/output invariance','event permutation','customer reindexing','gradient support'],
         'primitive_max_error':float((ref-actual).abs().max().detach()),'diagnostic':diagnostic,
         'versions':{p:importlib.metadata.version(p) for p in ['torch','pytorch-frame','torch-geometric']},
         'paper_benchmark':'NOT_RUN','paper_comparison':'INCOMPARABLE'}
 (ROOT/'_verify_l076_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='diagnostic'},indent=2));print('Diagnostic losses:',diagnostic['first_loss'],diagnostic['final_loss'])
if __name__=='__main__':run()
