"""Mutation checks on the exact temporal audit used by every model branch."""
import copy,json
from pathlib import Path
import torch
from torch_geometric.data import HeteroData
from relkit.batch_audit_l123 import audit_batch

def audit_check():
 g=HeteroData();g['site'].num_nodes=1;g['event'].num_nodes=3;g['event'].time=torch.tensor([4,8,12]);g['event','at','site'].edge_index=torch.tensor([[0,1,2],[0,0,0]])
 b=HeteroData();b['site'].n_id=torch.tensor([0,0]);b['site'].batch=torch.tensor([0,1]);b['site'].seed_time=torch.tensor([8,10]);b['site'].batch_size=2
 b['event'].n_id=torch.tensor([0,1,0,1]);b['event'].batch=torch.tensor([0,0,1,1]);b['event'].time=torch.tensor([4,8,4,8]);b['event','at','site'].edge_index=torch.tensor([[0,1,2,3],[0,0,1,1]]);b['event','at','site'].e_id=torch.tensor([0,1,0,1])
 rng=torch.get_rng_state();assert audit_batch(b,g,'site')['queries']==2;assert torch.equal(rng,torch.get_rng_state());rejected=[]
 for error in ['future_cutoff','timestamp_identity','cross_query','edge_identity']:
  bad=copy.deepcopy(b)
  if error=='future_cutoff':bad['site'].seed_time[0]=3
  if error=='timestamp_identity':bad['event'].time[0]=2
  if error=='cross_query':bad['event'].batch[0]=1
  if error=='edge_identity':bad['event','at','site'].e_id[0]=2
  try:audit_batch(bad,g,'site')
  except AssertionError:rejected.append(error)
  else:raise AssertionError(error)
 return dict(status='PASS',rejected=rejected,owner_cutoff_equality='PASS',audit_rng_unchanged=True)
if __name__=='__main__':
 r=audit_check();(Path(__file__).parent/'_audit_check_l153_results.json').write_text(json.dumps(r,indent=2));print(r)
