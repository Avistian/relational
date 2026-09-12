"""Identity must follow real class methods, nested helpers, defaults and weights."""
import json,subprocess,sys,types
from pathlib import Path
import torch
from relkit import openenv_l069_v2 as c
from relkit import tabpfn_l069_v2 as m
ROOT=Path(__file__).resolve().parent

def check():
 ns=vars(c);before=c.kernel_identity(ns,ROOT)['sha256'];old=c.shift_task
 exec('def helper69(x):return x+1\ndef shift_task(*a):return sum(helper69(x) for x in [1,2])',ns)
 first=c.kernel_identity(ns,ROOT)['sha256'];exec('def helper69(x):return x+2',ns);assert first!=c.kernel_identity(ns,ROOT)['sha256'];c.shift_task=old
 original=m.PackedAttention.forward
 def alternative(self,*a,**kw):return original(self,*a,**kw)
 m.PackedAttention.forward=alternative;assert c.kernel_identity(ns,ROOT)['sha256']!=before;m.PackedAttention.forward=original
 defaults=c.novelty_scores.__defaults__;c.novelty_scores.__defaults__=(.3,.7);assert c.kernel_identity(ns,ROOT)['sha256']!=before;c.novelty_scores.__defaults__=defaults
 exec('def callback69(x):return x+1\ndef shift_task(x,helper=callback69):return helper(x)',ns);first=c.kernel_identity(ns,ROOT)['sha256'];ns['callback69'].__defaults__=(2,);assert first!=c.kernel_identity(ns,ROOT)['sha256'];c.shift_task=old
 assert c.kernel_identity(ns,ROOT)['sha256']==before
 model,_=c.load_pretrained(c.ensure_checkpoint(ROOT));runtime=c.model_runtime_identity(model);weights=c.model_digest(model)
 model.blocks[0].feature.heads=3;assert c.model_runtime_identity(model)!=runtime;model.blocks[0].feature.heads=6
 hook=model.register_forward_hook(lambda *a:None)
 try:c.model_runtime_identity(model)
 except RuntimeError:pass
 else:raise AssertionError('Hook accepted')
 hook.remove()
 p=next(model.parameters());v=p.detach().clone()
 with torch.no_grad():p.add_(.01)
 assert c.model_digest(model)!=weights
 with torch.no_grad():p.copy_(v)
 code="import sys;sys.path.insert(0,'labs');from relkit import openenv_l069_v2 as c;print(c.kernel_identity(vars(c),'labs')['sha256'])"
 fresh=subprocess.check_output([sys.executable,'-c',code],cwd=ROOT.parent,text=True).strip();assert fresh==before,(before,fresh)
 return dict(status='PASS',identity=before,checks=['nested generator helper','actual PackedAttention.forward monkeypatch','numeric defaults','function-valued defaults','actual runtime head count','forward hook rejection','checkpoint weights','fresh process identity'])
if __name__=='__main__':
 r=check();(ROOT/'_identity_l069_v2_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
