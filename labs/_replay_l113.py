"""Independent full-data checkpoint replay with the pinned OGB class and PyG kernels."""
import ast,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv
from torch_sparse import SparseTensor
from relkit.scaling_l113 import file_hash

@torch.no_grad()
def replay(root,source_sha,seeds,preset="paper"):
 root=Path(root);blob=torch.load(root/'data/data.pt',weights_only=False);data=blob['data'];split=blob['split'];n=data.num_nodes
 # Leave duplicates uncoalesced: the released mean counts edge multiplicity.
 adj=SparseTensor(row=data.edge_index[1],col=data.edge_index[0],sparse_sizes=(n,n)).to('cuda')
 x=data.x.to('cuda');y=data.y.numpy();src=(Path('/work/sources/l113/cluster_gcn.py')).read_text()
 cls=next(a for a in ast.parse(src).body if isinstance(a,ast.ClassDef) and a.name=='SAGE');space={'torch':torch,'F':F,'SAGEConv':SAGEConv};exec(compile(ast.Module(body=[cls],type_ignores=[]),'pinned-release','exec'),space)
 results=[]
 for seed in seeds:
  path=root/source_sha/preset/f'seed-{seed}'
  if not (path/'result.json').exists():continue
  result=json.loads((path/'result.json').read_text())
  if result['status']!='COMPLETE':continue
  start=time.perf_counter();model=space['SAGE'](100,256,47,3,.5).to('cuda').eval();model.load_state_dict(torch.load(path/'checkpoint.pt',weights_only=True))
  pred=model(x,adj).argmax(-1).cpu().numpy();saved=np.load(path/'predictions.npz')['pred'];different=int(np.count_nonzero(pred!=saved))
  scores={k:float(np.mean(pred[idx.numpy()]==y[idx.numpy()])) for k,idx in split.items()}
  assert different/n<=1e-5,('Full original-model replay differs',seed,different)
  for k in scores:assert abs(scores[k]-result['selected'][k])<=1e-5
  row={'seed':seed,'status':'PASS','predictions':n,'class_disagreements':different,'scores':scores,'seconds':time.perf_counter()-start,'checkpoint_sha256':file_hash(path/'checkpoint.pt'),'backend':'Pinned original SAGE class, PyG SAGEConv + torch_sparse.SparseTensor, complete graph, no sampled edges'}
  (path/'replay.json').write_text(json.dumps(row,indent=2));results.append(row);del model
 return results
