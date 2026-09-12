"""Complete NumPy reconstruction of L069's historical model with compact attention.

The independent arithmetic is the immutable L064 NumPy operator. Original source
supplies checkpoint tensors, while the new live model supplies comparison outputs.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tabpfn.model.loading import load_model
from _independent_l064 import forward
from relkit import tabpfn_l069_v2 as live
ROOT=Path(__file__).resolve().parent

def check():
 torch.set_num_threads(1);path=ROOT/'data/cache/foundation/tabpfn-v2.ckpt';original,_,_=load_model(path=path,model_seed=0)
 state={k:v.detach().numpy().astype('float64') for k,v in original.state_dict().items()};model,_=live.load_pretrained(path);model.double();cases=[]
 for name,f in [('binary_even',4),('multiclass_odd',5),('missing_constant',3),('one_feature',1)]:
  rng=np.random.default_rng(690);x=rng.normal(size=(11,f));y=np.arange(6)%(2 if name=='binary_even' else 3)
  if name=='missing_constant':x[:,1]=4.;x[1,0]=np.nan;x[-1,2]=np.nan
  vectors=torch.randn(((f+1)//2,48),generator=torch.Generator().manual_seed(0),dtype=torch.float64).numpy();outputs=[];handles=[block.register_forward_hook(lambda m,a,v:outputs.append(v.detach().numpy().copy())) for block in model.blocks]
  with torch.no_grad():result=model(torch.from_numpy(x)[None],torch.from_numpy(y)[None])[0].numpy()
  for h in handles:h.remove()
  with threadpool_limits(limits=1):pred,layers,active=forward(x,y,state,vectors)
  delta=float(abs(pred-result).max());layer_delta=max(float(abs(a-b).max()) for a,b in zip(layers,outputs));assert max(delta,layer_delta)<1e-9,(name,delta,layer_delta)
  cases.append(dict(name=name,logit_delta=delta,all_layer_delta=layer_delta,active=active.tolist()))
 report=dict(status='PASS',core_sha256=hashlib.sha256((ROOT/'relkit/tabpfn_l069_v2.py').read_bytes()).hexdigest(),numpy_operator_sha256=hashlib.sha256((ROOT/'_independent_l064.py').read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cases=cases,scope='Independent NumPy encoding, all 12 feature/row attention and feed-forward blocks, every intermediate layer and 10-output head, compared with the actual L069 compact-attention model; no benchmark reproduction claim.')
 (ROOT.parent/'reviews/lesson-quality-audit-047-070/069-independent.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':print(json.dumps(check(),indent=2))
