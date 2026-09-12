"""Fresh process/live default/nested helper/runtime/weights binding before measurement."""
import json,sys,subprocess,types
from pathlib import Path
import torch
from relkit import driftpfn_l068_v2 as c
ROOT=Path(__file__).resolve().parent

def check():
 ns=vars(c);before=c.kernel_identity(ns)['sha256'];original=c.sample_scm
 exec('def nested68(x):return x+1\ndef sample_scm(*args,**kwargs):return sum(nested68(x) for x in [1,2])',ns)
 first=c.kernel_identity(ns)['sha256'];exec('def nested68(x):return x+2',ns);assert c.kernel_identity(ns)['sha256']!=first
 c.sample_scm=original;assert c.kernel_identity(ns)['sha256']==before
 d=c.temporal_split.__defaults__;c.temporal_split.__defaults__=(7,);assert c.kernel_identity(ns)['sha256']!=before;c.temporal_split.__defaults__=d
 exec('def callback68(x):return x+1\ndef sample_scm(seed=0,helper=callback68):return helper(seed)',ns)
 first=c.kernel_identity(ns)['sha256'];ns['callback68'].__defaults__=(2,);assert c.kernel_identity(ns)['sha256']!=first;c.sample_scm=original
 model,_=c.load_pretrained(ROOT);runtime=c.model_runtime_identity(model);weights=c.model_digest(model)
 model.blocks[0].feature.heads=3;assert c.model_runtime_identity(model)!=runtime;model.blocks[0].feature.heads=6
 p=next(model.parameters());v=p.detach().clone()
 with torch.no_grad():p.add_(.01)
 assert c.model_digest(model)!=weights
 with torch.no_grad():p.copy_(v)
 hook=model.register_forward_hook(lambda *a:None)
 try:c.model_runtime_identity(model)
 except RuntimeError:pass
 else:raise AssertionError('Hook bypass')
 hook.remove();model.forward=types.MethodType(lambda s,*a:None,model)
 try:c.model_runtime_identity(model)
 except RuntimeError:pass
 else:raise AssertionError('Instance override bypass')
 del model.forward
 assert c.model_runtime_identity(model)==runtime and c.model_digest(model)==weights
 script="import sys;sys.path.insert(0,'labs');from relkit import driftpfn_l068_v2 as c;print(c.kernel_identity(vars(c))['sha256'])"
 fresh=subprocess.check_output([sys.executable,'-c',script],cwd=ROOT.parent,text=True).strip();assert fresh==before
 return dict(status='PASS',identity=before,fresh_process_identity=fresh,checks=['nested generator helper','function-valued defaults','numeric defaults','actual weights','runtime settings','module hook','instance override','fresh process'])
if __name__=='__main__':
 r=check();(ROOT/'_identity_l068_v2_results.json').write_text(json.dumps(r,indent=2));print(r)
