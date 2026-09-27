"""Reject temporal mistakes and prove batch auditing preserves sampler RNG."""
import copy,json
from pathlib import Path
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from relkit.temporal_l123 import eligible,sample_temporal,audit_sample,fixture
from relkit.batch_audit_l123 import audit_batch
P=Path(__file__).resolve().parent
nodes,edges=fixture();good=sample_temporal(nodes,edges,('person',0),8,2);rejected=[]
for name,node,index in [('future_second_hop',('memo',0),3),('late_arrival',('transfer',2),2)]:
 bad=copy.deepcopy(good);bad['nodes'].add(node);bad['edges'].append(index)
 try:audit_sample(nodes,edges,bad)
 except ValueError:rejected.append(name)
 else:raise AssertionError(name)
g=HeteroData();g['person'].num_nodes=1;g['event'].num_nodes=4;g['event'].time=torch.tensor([4,8,9,12]);g['memo'].num_nodes=2;g['memo'].time=torch.tensor([7,12])
g['event','to','person'].edge_index=torch.tensor([[0,1,2,3],[0,0,0,0]])
g['memo','to','event'].edge_index=torch.tensor([[0,1],[0,0]])
b=next(iter(NeighborLoader(g,num_neighbors=[-1,-1],input_nodes=('person',torch.tensor([0,0])),input_time=torch.tensor([8,10]),time_attr='time',batch_size=2)))
b['person'].seed_time=torch.tensor([8,10]);before=torch.get_rng_state();audit_batch(b,g,'person');assert torch.equal(before,torch.get_rng_state())
assert set(b['event'].n_id[b['event'].batch==0].tolist())=={0,1}, 'Equality boundary must include t=8'
assert set(b['event'].n_id[b['event'].batch==1].tolist())=={0,1,2}
assert set(b['memo'].n_id.tolist())=={0}, 'Second hop uses root time, not parent time4'
for name in ['timestamp_identity','cross_query','edge_identity','root_cutoff']:
 bad=copy.deepcopy(b)
 if name=='timestamp_identity':bad['event'].time[0]=3
 if name=='cross_query':bad['event'].batch[0]=1
 if name=='edge_identity':bad['event','to','person'].e_id[0]=3
 if name=='root_cutoff':bad['person'].seed_time[0]=3
 try:audit_batch(bad,g,'person')
 except AssertionError:rejected.append(name)
 else:raise AssertionError(name)
# For fixed RNG, the auditing layer cannot change sampled nodes or edges.
def draw():
 return next(iter(NeighborLoader(g,num_neighbors=[1,1],input_nodes=('person',torch.tensor([0])),input_time=torch.tensor([10]),time_attr='time',batch_size=1)))
torch.manual_seed(123);plain=draw();torch.manual_seed(123);audited=draw();audited['person'].seed_time=torch.tensor([10]);audit_batch(audited,g,'person')
for k in plain.node_types:assert torch.equal(plain[k].n_id,audited[k].n_id)
for k in plain.edge_types:assert torch.equal(plain[k].edge_index,audited[k].edge_index)
r=dict(status='PASS',rejected=rejected,equality_boundary='PASS',root_vs_parent_cutoff='PASS',same_entity_mixed_times='PASS',audit_rng_unchanged=True,sampled_outputs_unchanged=True)
(P/'_mutation_l123_results.json').write_text(json.dumps(r,indent=2));print(r)
