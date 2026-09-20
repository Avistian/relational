"""Native all-neighbor parity on the real complete ACM graph, fixed probe seeds."""
import json
from pathlib import Path
import torch
from relkit.checkpoint_l100 import load_graph,audit_batches,ARMS
P=Path(__file__).resolve().parent;torch.set_num_threads(2)
g=load_graph(P/'data/l099');reports=[]
for arm in ARMS:
 for bs in [1,7,23]:
  r=audit_batches(g,arm,bs,seeds=g['train'][:23]);reports.append(r)
  print(arm,bs,r['max_logit_gap'],r['max_gradient_gap'],flush=True)
(P/'_batch_audit_l100_results.json').write_text(json.dumps({'status':'PASS','scope':'23 fixed training seeds on complete ACM graph, all neighbors; not exhaustive all-seed parity','tolerance':2e-6,'reports':reports},indent=2)+'\n')
