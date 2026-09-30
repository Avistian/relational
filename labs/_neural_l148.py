"""Counterfactual neural probes: forbidden information must not affect outputs."""
import json
from pathlib import Path
import torch,pandas as pd
from torch_frame import stype
from torch_frame.data import Dataset
from torch_geometric.data import HeteroData
from relkit.rdl_l117 import Model
from relkit.ablation_model_l148 import AblationModel,restrict_history

def run():
 torch.set_num_threads(1);data=HeteroData();stats={}
 for kind,values in [('drivers',[1.,2.,3.,4.]),('results',[3.,5.,9.,10.])]:
  d=Dataset(pd.DataFrame({'x':values}),col_to_stype={'x':stype.numerical}).materialize()
  data[kind].tf=d.tensor_frame;stats[kind]=d.col_stats;data[kind].batch=torch.tensor([0,1,0,1]);data[kind].n_id=torch.arange(4)
 data['drivers'].seed_time=torch.tensor([400,500])*86400
 data['results'].time=torch.tensor([34,136,40,140])*86400
 data['results','to','drivers'].edge_index=torch.tensor([[0,1,2,3],[0,1,0,1]])
 data['drivers','rev_to','results'].edge_index=data['results','to','drivers'].edge_index.flip(0)
 for kind in data.node_types:data[kind].num_sampled_nodes=[2,1,1]
 for edge in data.edge_types:data[edge].num_sampled_edges=[2,2]
 torch.manual_seed(148);full=AblationModel(data,stats,2,128,1,'sum','batch_norm',arm='full').eval()
 ref=Model(data,stats,2,128,1,'sum','batch_norm').eval();ref.load_state_dict(full.state_dict())
 torch.testing.assert_close(full(data,'drivers'),ref(data,'drivers'),atol=0,rtol=0)
 import copy
 result={};changed=copy.deepcopy(data);changed['results'].tf.feat_dict[stype.numerical][0]=9000
 for arm in ['full','encoder','messages','history','combined']:
  torch.manual_seed(148);m=AblationModel(data,stats,2,128,1,'sum','batch_norm',arm=arm).eval()
  before=m(data,'drivers');after=m(changed,'drivers')
  diff=float((before-after).abs().max().detach())
  if arm in ['messages','history','combined']:assert diff==0,(arm,diff)
  else:assert diff>1e-6,(arm,diff)
  if arm in ['encoder','combined']:assert all(len(e.backbone)==1 for e in m.encoder.encoders.values())
  m.zero_grad();before.sum().backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
  result[arm]=dict(changed_old_neighbor_output_max_difference=diff,parameters=sum(p.numel() for p in m.parameters()))
 pruned,n=restrict_history(data,'drivers');assert n==1 and pruned['results'].n_id.tolist()==[1,2,3]
 torch.testing.assert_close(pruned['drivers'].seed_time,data['drivers'].seed_time)
 assert pruned['results','to','drivers'].edge_index.tolist()==[[0,1,2],[1,0,1]]
 return dict(status='PASS',baseline_output_parity='EXACT_CPU_FIXTURE',counterfactuals=result,scope='Synthetic neural probes, not benchmark results')
if __name__=='__main__':
 r=run();Path(__file__).with_name('evidence').joinpath('l148/neural.json').write_text(json.dumps(r,indent=2));print(r)
