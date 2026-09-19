"""One full-width real-PPI update; timing/shape preflight, NOT a paper reproduction."""
import hashlib,json,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'relkit'))
from cluster_gcn_l089 import *
LAB=Path(__file__).resolve().parent
torch.set_num_threads(1);torch.manual_seed(1)
d=load_ppi(LAB/'data/l089');c=preset('paper');p=prepare(d,50);nodes=p['parts'][0]
a=p['a'][nodes][:,nodes].copy();a.data[:]=1
s=sparse_tensor(enhanced_support(a));x=torch.tensor(p['pre'][nodes]);y=torch.tensor(p['y'][nodes])
m=ClusterGCN();opt=ReleaseAdam(m.parameters());t=time.perf_counter()
z=m(x,s);loss=F.binary_cross_entropy_with_logits(z,y);loss.backward();opt.step()
assert z.shape==(len(nodes),121) and torch.isfinite(loss)
r={'status':'PASS','lane':'single_update_preflight_only','nodes':len(nodes),'layers':5,'width':2048,'parameters':sum(p.numel() for p in m.parameters()),'loss':float(loss.detach()),'update_seconds':time.perf_counter()-t,'device':'cpu','full_epochs':'NOT_RUN','source_sha256':hashlib.sha256((LAB/'relkit/cluster_gcn_l089.py').read_bytes()).hexdigest()}
(LAB/'_scale_l089_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
