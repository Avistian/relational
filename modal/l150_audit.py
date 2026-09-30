"""USD10 aggregate, immutable attempts, no automatic retries."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];app=modal.App('l150-relgnn')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html'))
for p in ['_full_l143.py','_parity_l143.py','relkit/reproduction_l143.py','relkit/relgnn_l143.py']:image=image.add_local_file(ROOT/'labs'/p,'/work/'+p)
image=image.add_local_dir(ROOT/'labs/sources/l141','/work/sources/l141')
volume=modal.Volume.from_name('l150-relgnn-evidence',create_if_missing=True)
RATE=.00022572

@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=600,retries=0,volumes={'/evidence':volume})
def inspect():
 import sys,json,time,torch,numpy as np
 sys.path.insert(0,'/work');from _full_l143 import get_dataset,get_task,CONFIG,get_atomic_routes,get_node_train_table_input,NeighborLoader
 from relkit.relgnn_l143 import RelGNN_Model
 from _parity_l143 import original_modules
 root=Path('/evidence');start=time.perf_counter();data,stats=torch.load(root/'prepared/graph.pt',weights_only=False);cp=torch.load(root/'prepared/released.pth',weights_only=False,map_location='cpu')
 dataset=get_dataset('rel-f1');dataset.cache_dir=str(root/'prepared/unpacked');db=dataset.get_db()
 task=get_task('rel-f1','driver-position');task.cache_dir=str(root/'prepared/unpacked/driver-position')
 print('QUALIFYING',data['qualifying'].tf.col_names_dict,flush=True)
 print('CP numerical stats',[(k,v.tolist()) for k,v in cp.items() if 'qualifying.encoder.encoder_dict.numerical' in k and ('mean' in k or 'std' in k)],flush=True)
 print('Raw columns',db.table_dict['qualifying'].df.describe(include='all').to_string(),flush=True)
 from torch_geometric.seed import seed_everything
 seed_everything(0);kwargs=dict(data=data,col_stats_dict=stats,out_channels=1,norm='batch_norm',atomic_routes=get_atomic_routes(data.edge_types),**CONFIG)
 a=RelGNN_Model(**kwargs).cuda()
 inp=get_node_train_table_input(task.get_table('train'),task)
 dl=NeighborLoader(data,num_neighbors=[128,64],time_attr='time',input_nodes=inp.nodes,input_time=inp.time,transform=inp.transform,subgraph_type='bidirectional',batch_size=512,temporal_strategy='uniform',shuffle=True)
 a.eval()
 with torch.no_grad():a(next(iter(dl)).cuda(),task.entity_table)
 bch=next(iter(dl)).cuda()
 with original_modules('/work/sources/l141') as (Original,_):
  b=Original(**kwargs).cuda();b.load_state_dict(a.state_dict());a.train();b.train()
  cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all()
  pa=a(bch,task.entity_table).view(-1)
  torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(gpu_rng)
  pb=b(bch,task.entity_table).view(-1)
  torch.testing.assert_close(pa,pb,rtol=1e-5,atol=1e-6)
  torch.nn.functional.l1_loss(pa,bch[task.entity_table].y).backward();torch.nn.functional.l1_loss(pb,bch[task.entity_table].y).backward()
  nonfinite={};err=0.
  for (n,p),(m,q) in zip(a.named_parameters(),b.named_parameters()):
   assert n==m
   if p.grad is None:assert q.grad is None;continue
   mask=torch.isfinite(p.grad);assert torch.equal(mask,torch.isfinite(q.grad))
   if not mask.all():nonfinite[n]=int((~mask).sum())
   if mask.any():err=max(err,float((p.grad[mask]-q.grad[mask]).abs().max()))
   torch.testing.assert_close(p.grad[mask],q.grad[mask],rtol=5e-4,atol=5e-5)
 report=dict(checkpoint_qualifying_numeric={k:v.tolist() for k,v in cp.items() if 'qualifying.encoder.encoder_dict.numerical' in k and ('mean' in k or 'std' in k)},fresh_qualifying_columns={str(k):v for k,v in data['qualifying'].tf.col_names_dict.items()},raw_numeric_means={k:float(v) for k,v in db.table_dict['qualifying'].df.select_dtypes('number').mean().items()},matched_nonfinite_gradients=nonfinite,max_finite_gradient_error=err,seconds=time.perf_counter()-start)
 (root/'diagnosis.json').write_text(json.dumps(report,indent=2));volume.commit();return report
@app.local_entrypoint()
def audit_main():
 from l150_repro import reserve
 reserve('diagnosis',seconds=600);print(inspect.remote())
