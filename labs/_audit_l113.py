"""Independent official-loader, partition and compact-label audit on prepared full data."""
import json,time
from pathlib import Path
import numpy as np
import torch
from ogb.io.read_graph_raw import read_csv_graph_raw
from torch_geometric.loader import ClusterLoader
from relkit.scaling_l113 import file_hash,tensor_hash,induced_edges

def audit(root):
 root=Path(root);start=time.perf_counter();blob=torch.load(root/'data.pt',weights_only=False);data=blob['data'];split=blob['split']
 official=read_csv_graph_raw(str(root/'products/raw'),add_inverse_edge=True)[0]
 assert np.array_equal(official['node_feat'],data.x.numpy())
 # Canonical sort allows differing raw-edge order, preserving multiplicities.
 n=data.num_nodes;expected=torch.from_numpy(official['edge_index']);a=(expected[0]*n+expected[1]).sort().values;b=(data.edge_index[0]*n+data.edge_index[1]).sort().values
 assert torch.equal(a,b);del a,b,expected,official
 groups=torch.load(root/'clusters.pt',weights_only=False);checks=[]
 for chosen in [[0,1],[3,4,5]]:
  batch=ClusterLoader(groups,batch_size=len(chosen))._collate(chosen)
  local=torch.cat([torch.arange(int(groups.partition.partptr[i]),int(groups.partition.partptr[i+1])) for i in chosen])
  ids=groups.partition.node_perm[local]
  assert torch.equal(batch.x,data.x[ids]) and torch.equal(batch.train_mask,data.train_mask[ids])
  exp=induced_edges(data.edge_index,ids,n);size=len(ids)
  assert torch.equal((exp[0]*size+exp[1]).sort().values,(batch.edge_index[0]*size+batch.edge_index[1]).sort().values)
  checks.append({'clusters':chosen,'nodes':len(ids),'edges':exp.shape[1]})
 np.savez_compressed(root/'labels-splits.npz',y=data.y.numpy().astype(np.uint8),**{k:v.numpy().astype(np.int32) for k,v in split.items()})
 r={'status':'PASS','official_reader':'ogb.io.read_graph_raw.read_csv_graph_raw','features':'EXACT','edge_multiset':'EXACT including duplicate multiplicity','cluster_induced_checks':checks,'seconds':time.perf_counter()-start,'labels_splits_sha256':file_hash(root/'labels-splits.npz')}
 (root/'audit.json').write_text(json.dumps(r,indent=2));return r
if __name__=='__main__':print(audit('labs/data/l113'))
