"""Execute pinned PyG source operators against the visible implementation."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import torch
from relkit.hetero_l133 import audit_layer
P=Path(__file__).resolve().parent
classes={};hashes={}
for name,cls in [('sage_conv.py','SAGEConv'),('hetero_conv.py','HeteroConv')]:
 p=P/'sources/l117/primitives'/name
 spec=importlib.util.spec_from_file_location('l133_'+p.stem,p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 classes[cls]=getattr(m,cls);hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(1);torch.manual_seed(133)
e={('a','r','b'):torch.tensor([[0,1,1],[0,0,2]]),('b','rev','a'):torch.tensor([[0,2],[0,1]])}
x={'a':torch.randn(2,3),'b':torch.randn(4,3)}
c=classes['HeteroConv']({k:classes['SAGEConv']((3,3),3,aggr='sum') for k in e},aggr='sum')
r=audit_layer(c,x,e);r.update(source_sha256=hashes,scope='Pinned PyG2.6.1 Python operators executed with current local dependencies; full pinned GPU runtime checked separately')
(P/'_source_check_l133_results.json').write_text(json.dumps(r,indent=2));print(r)
