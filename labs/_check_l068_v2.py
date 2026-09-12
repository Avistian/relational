"""Source preprocessing/data correspondence and live mechanism diagnostics."""
import ast,hashlib,json,sys,types,os
from pathlib import Path
import numpy as np,pandas as pd,torch
from relkit import driftpfn_l068_v2 as c
from _paper_audit_l068_v2 import original_model,OFFICIAL
ROOT=Path(__file__).resolve().parent

def check():
 torch.set_num_threads(1);original_model('dist')
 from tabpfn.utils import min_max_scale_data
 from tabpfn.model.encoders import Time2VecEncoderStep
 from tabpfn.scripts.model_builder import get_encoder
 errors=[]
 _,config=original_model('dist')
 for f in [1,2,3,6]:
  x=torch.arange(13*f,dtype=torch.float64).reshape(13,1,f)/10
  if f>1:x[:,:,0]=7
  x[2,0,-1]=float('nan')
  g=c.group_features(x.transpose(0,1)).permute(1,0,2,3).reshape(13,-1,2)
  enc=get_encoder(config)(2,192).double();state={'main':g.clone()}
  for step in list(enc)[:4]:state=step(state,single_eval_pos=7)
  err=float((state['main']-c.numeric_groups(g,7)).abs().max());assert err<1e-12;errors.append(err)
 for values,n in [([3,3,4,100],2),([0,2,4,5,40],3),([-4,-2,0,2],2)]:
  x=torch.tensor(values,dtype=torch.float64)[:,None,None]
  torch.testing.assert_close(c.normalize_time(x,n),min_max_scale_data(x,normalize_positions=n),rtol=0,atol=0)
 # Execute the original dataset functions without changing their computation.
 source=(OFFICIAL/'tabpfn/datasets/dist_shift_datasets.py').read_text();names=['dataframe_to_distribution_shift_ds','get_electricity_data','get_parking_birmingham_data','get_chess_data','get_intersecting_blobs']
 mod=types.ModuleType('tabpfn.datasets');mod.__path__=[]
 class Collector:
  def __init__(self,**kwargs):self.__dict__.update(kwargs)
 mod.DistributionShiftDataset=Collector;sys.modules['tabpfn.datasets']=mod
 ns=dict(__name__='tabpfn.datasets.dist_shift_datasets',__package__='tabpfn.datasets',np=np,pd=pd,torch=torch,os=os,MODULE_DIR=str(ROOT/'data/cache/l068-release/tabpfn/datasets'),TASK_TYPE_MULTICLASS='multiclass')
 for node in ast.parse(source).body:
  if isinstance(node,ast.FunctionDef) and node.name in names:exec(compile(ast.Module(body=[node],type_ignores=[]),str(OFFICIAL/'tabpfn/datasets/dist_shift_datasets.py'),'exec'),ns)
 datasets={}
 for name,fn in [('electricity','get_electricity_data'),('parking','get_parking_birmingham_data'),('chess','get_chess_data'),('blobs','get_intersecting_blobs')]:
  o=ns[fn]();a=c.load_dataset(ROOT,name)
  np.testing.assert_array_equal(a['x'],o.x.float().numpy());np.testing.assert_array_equal(a['y'],o.y.numpy());np.testing.assert_array_equal(a['c'],o.dist_shift_domain.numpy())
  assert a['features']==o.attribute_names
  datasets[name]=dict(shape=list(a['x'].shape),source_x_sha256=hashlib.sha256(a['x'].tobytes()).hexdigest(),all_rows_columns_labels_domains_equal=True)
 # Sparse mapping and nonlinearity must drive generated data, not only an explanatory picture.
 d=c.sample_scm(2);trace=d['trace'];w=np.array(trace['weights']);base=np.array(trace['base']);mapping=np.array(trace['edge_to_relation']);selected=np.isin(mapping,trace['selected'])
 assert np.all(w[:,~selected]==base[~selected]) and np.any(w[:,selected]!=base[selected])
 old=c.shifted_weights
 try:
  c.shifted_weights=lambda base,edge_to_relation,selected,shifts:np.broadcast_to(base,shifts.shape)
  changed=c.sample_scm(2)
 finally:c.shifted_weights=old
 assert not np.array_equal(changed['x'],d['x']) and not np.array_equal(changed['y'],d['y'])
 # Source labels are inputs; held-out labels cannot enter model function signature.
 split=c.temporal_split(d['c'],5,0,24)
 assert not set(split['train'])&set(split['id']) and d['c'][split['train']].max()<d['c'][split['ood']].min()
 return dict(status='PASS',numeric_group_errors=errors,time_fixtures=3,datasets=datasets,scm_mapping_and_live_data_mutation=True,split_disjoint=True)
if __name__=='__main__':
 r=check();(ROOT/'_check_l068_v2_results.json').write_text(json.dumps(r,indent=2));print(r)
