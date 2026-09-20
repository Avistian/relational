"""Audit data, implementation provenance and memory pruning independently of scores."""
import hashlib,importlib.util,json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('rgcn_l091',P/'relkit/rgcn_l091.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
torch.set_num_threads(1);manifest=json.loads((P/'_sources_l091.json').read_text())
for name,digest in manifest['source_files'].items():assert hashlib.sha256((P/'sources/l091'/name).read_bytes()).hexdigest()==digest
full=m.load_aifb(P/'data/l091',manifest,prune=False);pruned=m.load_aifb(P/'data/l091',manifest,prune=True)
assert full[-1]['node_order_sha256']==pruned[-1]['node_order_sha256'];assert torch.equal(full[1],pruned[1]);roots=torch.cat([full[2],full[3]])
torch.manual_seed(123);net=m.RGCN(8285,91,4)
f=net(full[0]);p=net(pruned[0]);torch.testing.assert_close(f[roots],p[roots],atol=1e-7,rtol=1e-5)
params=list(net.parameters())
gf=torch.autograd.grad(m.masked_loss(f,full[1],full[2]),params)
gp=torch.autograd.grad(m.masked_loss(p,full[1],full[2]),params)
for a,b in zip(gf,gp):torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-5)
raw=json.loads((P/'_paper_l091_results.json').read_text());assert len(raw['runs'])==10
for r in raw['runs']:
 assert r['epochs']==50 and len(r['train_loss'])==50 and r['bases']==0
 assert sum(a==b for a,b in zip(r['test_predictions'],r['test_labels']))==r['test_correct']
 assert r['test_total']==36 and r['test_ids']==full[3].tolist()
result={'status':'PASS','data':pruned[-1],'source_file_hashes':'PASS','unpruned_vs_pruned_root_logits':'PASS','unpruned_vs_pruned_training_gradients':'PASS','raw_prediction_score_recomputation':'PASS','historical_runtime':'NOT_RUN','historical_seed_and_order_identity':'INCOMPARABLE','implementation_sha256':hashlib.sha256((P/'relkit/rgcn_l091.py').read_bytes()).hexdigest()}
(P/'_audit_l091_results.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='data'})
