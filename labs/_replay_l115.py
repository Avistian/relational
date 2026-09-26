"""Full selected checkpoint replay using canonical and original released models."""
import json,time
from pathlib import Path
import numpy as np
import torch
from relkit.patterns_l115 import GCN,load_arxiv,normalized_adjacency,sha256
from _source_check_l115 import original,transfer
from _sparse_input_l115 import binary_for_reference
P=Path(__file__).resolve().parent;torch.set_num_threads(1);start=time.perf_counter()
x,edge,y,split,audit=load_arxiv(P/'data/l112');adj=normalized_adjacency(edge,len(y))
# Native CSR GCNConv adds loops; supply loop-free raw binary input.
binary=binary_for_reference(edge,len(y))
rows=[]
with torch.no_grad():
 for seed in range(10):
  d=P/'evidence/l115/paper'/f'seed-{seed}';cp=P/'results/l115/paper'/f'seed-{seed}'/'checkpoint.pt';record=json.loads((d/'result.json').read_text());assert sha256(cp)==record['checkpoint_sha256']
  model=GCN();model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True));model.eval()
  ref=original()(128,256,40,3,.5);transfer(model,ref);ref.eval()
  a=model(x,adj);b=ref(x,binary);torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-5)
  saved=np.load(d/'predictions.npz')['pred'];p=a.argmax(1).numpy();q=b.argmax(1).numpy()
  mismatch_local=int(np.count_nonzero(p!=saved));mismatch_source=int(np.count_nonzero(q!=saved));assert mismatch_local==mismatch_source==0
  row={'seed':seed,'nodes':len(y),'original_max_log_probability_error':float((a-b).abs().max()),'canonical_vs_saved_mismatches':mismatch_local,'original_vs_saved_mismatches':mismatch_source};rows.append(row);print(row,flush=True)
r={'status':'PASS','runs':rows,'nodes_replayed':len(y)*10,'seconds':time.perf_counter()-start,'scope':'All final predictions via original GCN class and GCNConv, native CSR input; current CPU runtime','historical_identity':'NOT_ESTABLISHED'};(P/'_replay_l115_results.json').write_text(json.dumps(r,indent=2))
