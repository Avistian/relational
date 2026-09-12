"""Executable evidence-binding regressions for the live L067 implementation."""
from pathlib import Path
import json,subprocess,sys
import torch
from relkit import localpfn_l067_v2 as c
ROOT=Path(__file__).resolve().parent

def check():
 ns=vars(c);before=c.kernel_identity(ns,ROOT)['sha256'];old=c.run_experiment
 exec('def nested67(x): return x+1\ndef run_experiment(*args,**kwargs): return sum(nested67(x) for x in [1,2])',ns)
 first=c.kernel_identity(ns,ROOT)['sha256'];exec('def nested67(x): return x+2',ns);assert c.kernel_identity(ns,ROOT)['sha256']!=first
 c.run_experiment=old;assert c.kernel_identity(ns,ROOT)['sha256']==before
 olddefaults=c.local_normalize.__defaults__;c.local_normalize.__defaults__=(99,);assert c.kernel_identity(ns,ROOT)['sha256']!=before;c.local_normalize.__defaults__=olddefaults
 # Function-valued default normalization has no process-specific repr address.
 def default67(x):return x+1
 exec('def run_experiment(root,model,config=None,namespace=None,helper=None): return helper(1)',ns);c.run_experiment.__defaults__=(None,None,default67)
 a=c.kernel_identity(ns,ROOT)['sha256'];default67.__defaults__=(2,);assert c.kernel_identity(ns,ROOT)['sha256']!=a
 c.run_experiment=old;assert c.kernel_identity(ns,ROOT)['sha256']==before
 torch.set_num_threads(1);model,_=c.load_pretrained(c.ensure_checkpoint(ROOT));runtime=c.model_runtime_identity(model);weights=c.model_digest(model)
 p=next(model.parameters());saved=p.detach().clone()
 with torch.no_grad():p.add_(.01)
 assert c.model_digest(model)!=weights
 with torch.no_grad():p.copy_(saved)
 hook=p.register_hook(lambda g:g)
 try:c.model_runtime_identity(model)
 except RuntimeError:pass
 else:raise AssertionError('Missed parameter gradient hook')
 hook.remove();hook=model.register_full_backward_hook(lambda *args:None)
 try:c.model_runtime_identity(model)
 except RuntimeError:pass
 else:raise AssertionError('Missed module backward hook')
 hook.remove();old_eps=model.blocks[0].norm1.eps;model.blocks[0].norm1.eps=.01;assert c.model_runtime_identity(model)!=runtime;model.blocks[0].norm1.eps=old_eps
 assert c.model_runtime_identity(model)==runtime and c.model_digest(model)==weights
 script="import sys;sys.path.insert(0,'labs');from pathlib import Path;from relkit import localpfn_l067_v2 as c;print(c.kernel_identity(vars(c),Path('labs').resolve())['sha256'])"
 fresh=subprocess.check_output([sys.executable,'-c',script],cwd=ROOT.parent,text=True).strip();assert fresh==before,(fresh,before)
 return dict(status='PASS',kernel_identity=before,fresh_process_identity=fresh,checks=['nested generator helper','function-valued defaults','numeric defaults','actual weights','LayerNorm runtime','parameter gradient hook','module backward hook','fresh process'])
if __name__=='__main__':
 r=check();(ROOT/'_identity_l067_v2_results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
