"""Check every plotted grid probability against the untouched released models."""
from pathlib import Path
import contextlib,io,json
import numpy as np
import torch
from _evidence_l068 import original_environment,digest
ROOT=Path(__file__).resolve().parent
def check():
 torch.set_num_threads(1);path=ROOT/'_boundary_l068_v2_results.json';r=json.loads(path.read_text());load,ns=original_environment();data=ns['get_intersecting_blobs']();x=data.x.float().numpy();y=data.y.numpy();c=data.dist_shift_domain.numpy();context=np.flatnonzero(c<4);assert context.tolist()==r['context_ids'];grid=np.array(r['grid'],dtype=np.float32)
 gx,gy=np.meshgrid(np.linspace(-20,20,36),np.linspace(-26,17,36));np.testing.assert_array_equal(grid,np.column_stack([gx.ravel(),gy.ravel()]).astype(np.float32));assert len(r['records'])==6
 models={};checks=[]
 for row in r['records']:
  family='base' if row['arm']=='base_time' else 'dist'
  if family not in models:
   with contextlib.redirect_stdout(io.StringIO()):loaded,_=load(str(ROOT/f'data/cache/l068-release/tabpfn/model_cache/tabpfn_{family}_model_1.cpkt'),'cpu',verbose=False)
   models[family]=loaded[2].eval()
  model=models[family];model.generator_device=torch.device('cpu');model.generator.manual_seed(17);raw=np.r_[x[context],grid];times=np.r_[c[context],np.full(len(grid),row['domain'])].astype(np.float32)
  if family=='base':raw=np.column_stack([raw,times])
  inputs={'main':torch.from_numpy(raw)[:,None]}
  if family=='dist':inputs['dist_shift_domain']=torch.from_numpy(times)[:,None,None]
  with torch.no_grad():p=model((inputs,torch.tensor(y[context],dtype=torch.float32)[:,None]),single_eval_pos=len(context))[:,0,:3].softmax(-1).numpy()
  saved=np.array(row['probabilities']);delta=float(np.max(np.abs(p-saved)));assert delta<2e-4;assert abs(saved.max(1).mean()-row['mean_confidence'])<1e-6
  checks.append(dict(arm=row['arm'],domain=row['domain'],predictions=len(grid),max_probability_error=delta))
 report=dict(status='PASS',scope='All six plotted grids independently reconstructed with original data, fixed context and untouched source checkpoints; grid confidence is not calibration.',evidence_sha256=digest(path),checker_sha256=digest(__file__),records=checks)
 (ROOT.parent/'reviews/lesson-quality-audit-047-070/068-boundary.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':print(check())
