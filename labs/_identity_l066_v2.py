"""Reject stale inference identity after instance methods/settings/weights change."""
import json,types
from pathlib import Path
import torch
from relkit import tabicl_l066_v2 as core
ROOT=Path(__file__).resolve().parent

def check():
 torch.set_num_threads(1);model=core.load_pretrained(core.ensure_checkpoint(ROOT));base=core.model_runtime_identity(model)
 original=model.forward
 def changed(self,*args,**kwargs):return original(*args,**kwargs)*0
 model.forward=types.MethodType(changed,model)
 try:core.model_runtime_identity(model)
 except RuntimeError:pass
 else:raise AssertionError('Instance forward override must invalidate evidence')
 del model.forward
 norm=model.row_interactor.out_ln;eps=norm.eps;norm.eps=.01
 assert core.model_runtime_identity(model)!=base,'Mutable normalization configuration must invalidate evidence'
 norm.eps=eps
 assert core.model_runtime_identity(model)==base
 digest=core.model_digest(model)
 with torch.no_grad():next(model.parameters()).add_(.1)
 assert core.model_digest(model)!=digest
 result=dict(status='PASS',instance_forward=True,layernorm_eps=True,actual_weights=True,restored_runtime_sha256=base)
 (ROOT/'_identity_l066_v2_results.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':print(check())
