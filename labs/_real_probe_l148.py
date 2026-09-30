"""Real batch intervention audit and first-batch gradient diagnostic."""
from pathlib import Path
import json,numpy as np,torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from relkit.rdl_l117 import get_node_train_table_input
from relkit.ablation_model_l148 import AblationModel,restrict_history

def run(root):
 torch.set_num_threads(1);seed_everything(148);root=Path(root);g=torch.load(root/'prepared/graph.pt',weights_only=False);data,stats,task=g['data'],g['stats'],g['task']
 inp=get_node_train_table_input(task.get_table('train'),task)
 loader=NeighborLoader(data,num_neighbors=[128,64],time_attr='time',input_nodes=inp.nodes,input_time=inp.time,transform=inp.transform,batch_size=512,temporal_strategy='uniform',shuffle=False,num_workers=0)
 b=next(iter(loader));cutoffs=b[task.entity_table].seed_time;arrays={'cutoffs':cutoffs.numpy()};pruned,removed=restrict_history(b,task.entity_table);checks={}
 for kind in b.node_types:
  n=b[kind].num_nodes;roots=np.zeros(n,dtype=bool)
  if kind==task.entity_table:roots[:len(cutoffs)]=True
  arrays[kind+'_times']=b[kind].time.numpy() if 'time' in b[kind] else np.zeros(n,dtype=np.int64)
  arrays[kind+'_owners']=b[kind].batch.numpy();arrays[kind+'_undated']=np.full(n,'time' not in b[kind]);arrays[kind+'_roots']=roots
  if 'time' in pruned[kind]:
   t=pruned[kind].time;upper=cutoffs[pruned[kind].batch];assert (t<=upper).all()
   if kind!=task.entity_table:assert (t>=upper-365*86400).all()
 for (src,rel,dst),edge in pruned.edge_index_dict.items():
  if edge.numel():assert int(edge[0].max())<pruned[src].num_nodes and int(edge[1].max())<pruned[dst].num_nodes
 for arm in ['full','encoder','messages','history','combined']:
  seed_everything(148);m=AblationModel(data,stats,2,128,1,'sum','batch_norm',arm=arm).cuda();batch=b.to('cuda');pred=m(batch,task.entity_table).view(-1);loss=(pred-batch[task.entity_table].y).abs().mean();assert torch.isfinite(loss);loss.backward()
  checks[arm]=dict(loss=float(loss.detach()),nonfinite_gradient_entries=sum(int((~torch.isfinite(p.grad)).sum()) for p in m.parameters() if p.grad is not None),parameters=sum(p.numel() for p in m.parameters()))
 np.savez_compressed(root/'sampled-history.npz',**arrays)
 report=dict(status='PASS',sampled_nodes=sum(b.num_nodes_dict.values()),removed=removed,gradient_probe=checks,gradient_scope='One real initial batch; source missing-value behavior preserved, not a guarantee of all-step gradient health')
 (root/'real-probe.json').write_text(json.dumps(report,indent=2));return report
