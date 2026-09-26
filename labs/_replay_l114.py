"""Fresh all-ten original-model inference replay; no GCN retraining."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from relkit.ogb_l112 import GCN,load_arxiv,normalized_adjacency,sha256
from relkit.error_l114 import MLP
from _source_check_l112 import original as original_gcn,transfer
from _source_check_l114 import original as original_mlp
from _sparse_input_l112 import binary_for_reference
P=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--model',choices=['gcn','mlp'],required=True);p.add_argument('--evidence-root',type=Path);p.add_argument('--checkpoint-root',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();assert not (a.evidence_root or a.checkpoint_root) or a.output, 'Custom replay requires a separate --output record';torch.set_num_threads(1);start=time.perf_counter()
x,edge,y,split,audit=load_arxiv(P/'data/l112')
if a.model=='gcn':adj=normalized_adjacency(edge,len(y));binary=binary_for_reference(edge,len(y))
rows=[];origin='l112' if a.model=='gcn' else 'l114'
with torch.no_grad():
 for seed in range(10):
  d=(a.evidence_root or P/f'evidence/{origin}/paper')/f'seed-{seed}';cp=(a.checkpoint_root or P/f'results/{origin}/paper')/f'seed-{seed}'/'checkpoint.pt';record=json.loads((d/'result.json').read_text());assert sha256(cp)==record['checkpoint_sha256'];assert sha256(d/'predictions.npz')==record['predictions_sha256']
  if a.model=='gcn':model=GCN();ref=original_gcn()(128,256,40,3,.5)
  else:model=MLP();ref=original_mlp()(128,256,40,3,.5)
  model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True));model.eval()
  if a.model=='gcn':transfer(model,ref)
  else:ref.load_state_dict(model.state_dict())
  ref.eval()
  if a.model=='gcn':aa=model(x,adj);bb=ref(x,binary)
  else:aa=model(x);bb=ref(x)
  torch.testing.assert_close(aa,bb,rtol=2e-5,atol=2e-5)
  saved=np.load(d/'predictions.npz')['pred'];p1=aa.argmax(1).numpy();p2=bb.argmax(1).numpy()
  mismatches=int(np.count_nonzero(p1!=saved));source_mismatches=int(np.count_nonzero(p2!=saved));assert mismatches==source_mismatches==0
  row={'seed':seed,'checkpoint_sha256':sha256(cp),'predictions_sha256':sha256(d/'predictions.npz'),'nodes':len(y),'original_max_log_probability_error':float((aa-bb).abs().max()),'canonical_vs_saved_mismatches':mismatches,'original_vs_saved_mismatches':source_mismatches};rows.append(row);print(row,flush=True)
r={'status':'PASS','model':a.model,'training_origin':str(a.evidence_root) if a.evidence_root else origin,'runs':rows,'nodes_replayed':len(y)*10,'seconds':time.perf_counter()-start,'scope':'Fresh complete CPU inference with canonical and original classes; modern backend','historical_identity':'NOT_ESTABLISHED'};(a.output or P/f'_replay_l114_{a.model}_results.json').write_text(json.dumps(r,indent=2))
