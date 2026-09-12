"""Full copied-checkpoint stage and numeric-wrapper parity; live namespace accepted."""
import os,sys,json,subprocess,tempfile,hashlib
from pathlib import Path
import numpy as np
import torch
from _prepare_l066_reference import prepare
ROOT=Path(__file__).resolve().parent

def check(namespace=None,save=True):
 from relkit import tabicl_l066_v2 as core
 ns=vars(core) if namespace is None else namespace
 torch.set_num_threads(1);model=ns['load_pretrained'](ns['ensure_checkpoint'](ROOT))
 sources=json.loads((ROOT/'_sources_l066_v2.json').read_text())
 for name,sha in sources['files'].items():assert hashlib.sha256((ROOT/'sources/l066-v2'/name).read_bytes()).hexdigest()==sha
 with tempfile.TemporaryDirectory(prefix='l066-source-') as folder:
  dest=Path(folder)/'reference.pt';env=os.environ.copy();env['PYTHONPATH']=str(prepare(ROOT))+os.pathsep+str(ROOT);env['OMP_NUM_THREADS']='1'
  subprocess.run([sys.executable,str(ROOT/'_reference_l066_v2.py'),str(ns['ensure_checkpoint'](ROOT)),str(dest)],env=env,check=True)
  ref=torch.load(dest,weights_only=False)
 errors=[]
 with torch.no_grad():
  for fixture in ref['fixtures']:
   x,y=fixture['x'],fixture['y'];trace=model(x,y,return_stages=True)
   for key in ['column','row','logits']:
    error=float((trace[key]-fixture[key]).abs().max());errors.append(dict(shape=list(x.shape),stage=key,max_abs=error))
    torch.testing.assert_close(trace[key],fixture[key],atol=3e-4,rtol=3e-4)
  w=ref['wrapper'];p,trace=ns['predict_numeric'](model,w['x'][:27],w['y'],w['x'][27:])
  np.testing.assert_allclose(trace['preprocessed'],w['preprocessed'],atol=1e-6,rtol=1e-6)
  np.testing.assert_allclose(p,w['probabilities'],atol=2e-5,rtol=2e-5)
  base=ref['fixtures'][0];x,y=base['x'],base['y'];a=model(x,y);alter=x.clone();alter[:,-1]=123
  b=model(alter,y);torch.testing.assert_close(a[:,:-1],b[:,:-1],atol=2e-5,rtol=2e-5)
  xp=torch.cat([x,torch.randn(1,2,x.shape[-1])],1);torch.testing.assert_close(a,model(xp,y)[:,:a.shape[1]],atol=2e-5,rtol=2e-5)
 result=dict(status='PASS',errors=errors,wrapper_probability_max_abs=float(np.max(np.abs(p-w['probabilities']))),kernel_identity=ns['kernel_identity'](ns,ROOT),weights_sha256=ns['model_digest'](model),runtime_sha256=ns['model_runtime_identity'](model),checkpoint_sha256=core.CHECKPOINT_SHA,reference_worker_sha256=hashlib.sha256((ROOT/'_reference_l066_v2.py').read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256((ROOT/'_sources_l066_v2.json').read_bytes()).hexdigest())
 if save:(ROOT/'_check_l066_v2_results.json').write_text(json.dumps(result,indent=2)+'\n')
 return result
if __name__=='__main__':print(json.dumps(check(),indent=2))
