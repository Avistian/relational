"""Independent reconstruction of the saved second-order SCM diagnostics."""
import json,sys,numpy as np,torch,contextlib,io
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from _evidence_l068 import original_environment,split_rows,metric,digest
root=Path(__file__).resolve().parent;r=json.loads((root/'_verify_l068_v2_results.json').read_text());load,ns=original_environment();torch.set_num_threads(1);models={};checks=[]
for item in r['scm_diagnostics']:
 t=item['trace'];seed=item['seed'];rng=np.random.default_rng(seed)
 base=rng.normal(0,1,12);a=rng.normal(size=(1,5));b=rng.normal(size=5);w=rng.normal(size=(5,4));v=rng.normal(size=(4,12));noise=rng.normal(size=(8,96,8))
 for key,value in [('base',base),('second_order_a',a),('second_order_b',b),('second_order_w',w),('second_order_v',v)]:np.testing.assert_array_equal(value,t[key])
 domains=np.linspace(0,1.75,8);shared=np.tanh(domains[:,None]@a+b);hidden=np.sin(shared@w);shifts=.6*(hidden@v)
 mapping=np.array([0,0,1,1,-1,-1,2,2,3,3,-1,-1]);weights=base+shifts*np.isin(mapping,[0,3]);np.testing.assert_allclose(weights,t['weights'],rtol=0,atol=1e-14)
 src=[0,0,1,1,2,3,0,0,4,4,5,6];dst=[2,3,2,3,4,4,5,6,5,6,7,7];values=np.zeros_like(noise);values[:,:,:2]=noise[:,:,:2]
 for node in range(2,8):
  result=np.zeros((8,96))
  for edge in range(12):
   if dst[edge]==node:result+=weights[:,edge,None]*values[:,:,src[edge]]
  result+=.15*noise[:,:,node];values[:,:,node]=np.tanh(result) if node in [2,3,5,6] else result
 np.testing.assert_allclose(values[:,0],t['sample_nodes'],rtol=0,atol=1e-14)
 x=values[:,:,[0,1,4]].reshape(-1,3).astype(np.float32);y=(values[:,:,7]>0).astype(int).ravel();c=np.repeat(np.arange(8),96);np.testing.assert_array_equal(y.reshape(8,96).mean(1),t['class_fraction'])
 split=split_rows(c,5,seed,32);context=split['train'];query=split['ood'];assert item['context_ids']==context.tolist() and item['query_ids']==query.tolist() and item['targets']==y[query].tolist()
 arm=item['arm'];family='base' if arm=='base_time' else 'dist';ck=r['config']['checkpoints'][0]
 if family not in models:
  with contextlib.redirect_stdout(io.StringIO()):loaded,_=load(str(root.resolve()/f'data/cache/l068-release/tabpfn/model_cache/tabpfn_{family}_model_{ck}.cpkt'),'cpu',verbose=False)
  models[family]=loaded[2].eval()
 model=models[family];model.generator_device=torch.device('cpu');model.generator.manual_seed(17+seed);ids=np.r_[context,query];raw=x[ids];times=c[ids].astype(np.float32)
 if arm=='base_time':raw=np.column_stack([raw,times])
 inputs={'main':torch.from_numpy(raw)[:,None]}
 if family=='dist':inputs['dist_shift_domain']=torch.from_numpy(times)[:,None,None]
 with torch.no_grad():p=model((inputs,torch.tensor(y[context],dtype=torch.float32)[:,None]),single_eval_pos=len(context))[:,0,:2].softmax(-1).numpy()
 delta=float(np.max(np.abs(p-np.array(item['probabilities']))));assert delta<2e-4
 for key,value in metric(y[query],item['probabilities']).items():assert abs(value-item[key])<2e-6
 checks.append(dict(seed=seed,arm=arm,predictions=len(query),max_probability_error=delta))
report=dict(status='PASS',checker_sha256=digest(__file__),records=checks,evidence_sha256=digest(root/'_verify_l068_v2_results.json'),scope='Independent reconstruction of every second-order trajectory, functional edge map, row noise, observed node and target, split and original pretrained prediction in the six local SCM diagnostics; not original pretraining sampler parity.')
(root.parent/'reviews/lesson-quality-audit-047-070/068-scm.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
