"""Compare the raw loader to OGB's independent library-agnostic reader."""
import json,zipfile
from pathlib import Path
import numpy as np
import torch
from ogb.nodeproppred import NodePropPredDataset
from relkit.debug_l116 import load_arxiv,normalized_adjacency
P=Path(__file__).resolve().parent;root=P/'data/l112';torch.set_num_threads(1)
x,edge,y,split,audit=load_arxiv(root)
# Extract locally under the official expected directory, avoiding its HTTP prompt.
dest=root/'official/ogbn_arxiv';dest.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(root/'arxiv.zip') as z:
 for member in z.infolist():
  if member.is_dir():continue
  rel=Path(member.filename).relative_to('arxiv');target=dest/rel
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(member))
(dest/'RELEASE_v1.txt').touch()
(dest/'processed').mkdir(exist_ok=True)
# Regenerate this script's own cache; avoid loading an old pickled runtime object.
(dest/'processed/data_processed').unlink(missing_ok=True)
reference=NodePropPredDataset(name='ogbn-arxiv',root=str(root/'official'))
graph,labels=reference[0];official=reference.get_idx_split()
assert np.array_equal(graph['node_feat'],x.numpy()) and np.array_equal(graph['edge_index'],edge.numpy())
assert np.array_equal(labels.reshape(-1),y.numpy())
for k,v in split.items():assert np.array_equal(v.numpy(),official[k])
adj=normalized_adjacency(edge,len(y));audit['normalized_nonzero']=adj.values().numel()
audit['status']='PASS';audit['official_loader_arrays']='EXACT';audit['label_visibility']='Only train labels contribute to loss; full graph and all node features visible'
(P/'_audit_l116_results.json').write_text(json.dumps(audit,indent=2));print(audit)
