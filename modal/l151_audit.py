"""Independent real-batch source/gradient differential at seed0 selected weights."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html'))
image=image.add_local_file(ROOT/'labs/relkit/rdl_l117.py','/work/relkit/rdl_l117.py').add_local_dir(ROOT/'labs/sources/l151','/work/sources/l151')
volume=modal.Volume.from_name('l151-portfolio-evidence')
RATE=.00033228
app=modal.App('l151-gradient-audit')
@app.function(image=image,gpu='T4',cpu=2,memory=65536,timeout=600,retries=0,volumes={'/evidence':volume})
def audit():
 import os,sys,json,time,traceback,copy,importlib.util
 os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');sys.path.insert(0,'/work')
 import torch
 from torch_geometric.loader import NeighborLoader
 from torch_geometric.seed import seed_everything
 from relbench.base import Table
 from relbench.tasks import get_task
 from relkit.rdl_l117 import Model,get_node_train_table_input
 root=Path('/evidence');assert not(root/'gradient-started').exists();(root/'gradient-started').touch();volume.commit();start=time.perf_counter()
 try:
  seed_everything(151);torch.set_num_threads(1)
  data,stats=torch.load(root/'prepared/graph.pt',weights_only=False)
  table=Table.load(root/'prepared/unpacked/study-outcome/train.parquet');task=get_task('rel-trial','study-outcome');q=get_node_train_table_input(table,task)
  loader=NeighborLoader(data,num_neighbors=[64,32],time_attr='time',input_nodes=q.nodes,input_time=q.time,transform=q.transform,batch_size=512,temporal_strategy='uniform',shuffle=False,num_workers=0)
  batch=next(iter(loader)).to('cuda')
  spec=importlib.util.spec_from_file_location('source_model','/work/sources/l151/examples__model.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  models=[Model(data,stats,2,128,1,'mean','batch_norm').cuda(),m.Model(data,stats,2,128,1,'mean','batch_norm').cuda()]
  state=torch.load(root/'ref-0/selected.pt',weights_only=False);gradients=[];outputs=[]
  for model in models:
   model.load_state_dict(state);model.train();seed_everything(151)
   pred=model(copy.deepcopy(batch),'studies').view(-1);loss=torch.nn.functional.binary_cross_entropy_with_logits(pred,batch['studies'].y.float());loss.backward()
   outputs.append(pred.detach().cpu());gradients.append({n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None})
  torch.testing.assert_close(outputs[0],outputs[1],rtol=1e-5,atol=1e-6)
  assert gradients[0].keys()==gradients[1].keys();max_error=0.;nonfinite={}
  for name,left in gradients[0].items():
   right=gradients[1][name];assert torch.equal(torch.isfinite(left),torch.isfinite(right)),name
   mask=torch.isfinite(left);count=int((~mask).sum())
   if count:nonfinite[name]=count
   if mask.any():
    delta=float((left[mask]-right[mask]).abs().max());max_error=max(max_error,delta);torch.testing.assert_close(left[mask],right[mask],rtol=1e-4,atol=2e-6)
  r=dict(status='PASS',scope='One real training batch at seed0 selected checkpoint; source agreement is not gradient health',roots=512,gradient_tensors=len(gradients[0]),matched_nonfinite_gradients=nonfinite,max_finite_gradient_error=max_error,max_output_error=float((outputs[0]-outputs[1]).abs().max()))
  (root/'gradient-audit.json').write_text(json.dumps(r,indent=2));return r
 except Exception:
  (root/'gradient-failure.txt').write_text(traceback.format_exc());raise
 finally:
  (root/'gradient-cost.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,worker_body_usd=(time.perf_counter()-start)*RATE)));volume.commit()
@app.local_entrypoint()
def main():
 from l151_repro import reserve
 reserve('gradient-audit-packaging-fix',seconds=600);print(audit.remote())
