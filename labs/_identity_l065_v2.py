"""Independent mutation and fresh-process checks of L065 live identities."""
import hashlib,json,subprocess,sys
from pathlib import Path
import torch
from relkit import query_embeddings_l065_v2 as core,tabpfn_l064_v2 as back
from relkit.data import load_tier_a
root=Path(__file__).resolve().parent;ns=vars(core)
base=core.kernel_identity(ns,root)['sha256'];checks={}
original=core.extract_embeddings.__defaults__;core.extract_embeddings.__defaults__=(1,None)
checks['positional_default']=core.kernel_identity(ns,root)['sha256']!=base;core.extract_embeddings.__defaults__=original
original=core.extract_embeddings.__kwdefaults__;core.extract_embeddings.__kwdefaults__={'probe':1}
checks['keyword_default']=core.kernel_identity(ns,root)['sha256']!=base;core.extract_embeddings.__kwdefaults__=original
core.PROTOCOL['folds']=9;checks['protocol']=core.kernel_identity(ns,root)['sha256']!=base;core.PROTOCOL['folds']=10
specs=load_tier_a.__globals__['SPECS'];specs['_l065_probe']={'value':1};checks['loader_specs']=core.kernel_identity(ns,root)['sha256']!=base;del specs['_l065_probe']
cache=load_tier_a.__globals__['CACHE'];load_tier_a.__globals__['CACHE']=Path('/tmp/changed-l065-cache');checks['loader_cache']=core.kernel_identity(ns,root)['sha256']!=base;load_tier_a.__globals__['CACHE']=cache
original=back.V2Block.forward
def changed_forward(self,*a,**k):return original(self,*a,**k)
back.V2Block.forward=changed_forward;checks['model_method']=core.kernel_identity(ns,root)['sha256']!=base;back.V2Block.forward=original
assert core.kernel_identity(ns,root)['sha256']==base
fresh=subprocess.check_output([sys.executable,'-c',"from pathlib import Path;from relkit import query_embeddings_l065_v2 as c;print(c.kernel_identity(vars(c),Path('labs').resolve())['sha256'])"],text=True).strip();checks['fresh_process_stable']=fresh==base
model,_=core.load_pretrained(core.ensure_checkpoint(root));digest=core.model_digest(model);parameter=next(model.parameters());old=parameter.detach().clone()
with torch.no_grad():parameter.add_(1)
checks['weight_edit']=core.model_digest(model)!=digest
with torch.no_grad():parameter.copy_(old)
assert core.model_digest(model)==digest
assert all(checks.values()),checks
result=dict(status='PASS',kernel_sha256=base,checks=checks,source_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest())
(root/'_identity_l065_v2_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
