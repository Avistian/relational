"""Full selected-checkpoint original-model replay; restores each snapshot's random state."""
import argparse,json,sys,hashlib,time,platform,importlib.metadata,ast
from pathlib import Path
import numpy as np,torch
P=Path(__file__).resolve().parent;sys.path[:0]=[str(P/'relkit'),str(P/'sources/l107/original')]
import sbm_l107 as s,egcn_h,egcn_o,models,utils
from scipy.sparse import coo_matrix
from sklearn.metrics import average_precision_score
from _analyze_l107 import ap,mrr

def original_metric_logger():
 # Compile the three unedited metric methods, avoiding unrelated plotting imports.
 # np.float was removed after the historical release; its old meaning was float.
 np.float=float
 tree=ast.parse((P/'sources/l107/original/logger.py').read_text())
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Logger')
 cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['get_MRR','get_row_MRR','get_MAP']]
 scope={'np':np,'torch':torch,'coo_matrix':coo_matrix,'average_precision_score':average_precision_score}
 exec(compile(ast.Module(body=[cls],type_ignores=[]),'original/logger.py','exec'),scope)
 return scope['Logger']()


def replay(root,data,variant,device):
 root=Path(root);d=s.load_sbm(data);saved=torch.load(root/'best.pt',map_location=device,weights_only=False)
 hidden=50 if variant=='H' else 51;cls=100 if variant=='H' else 565
 args=utils.Namespace({'feats_per_node':d['f'],'layer_1_feats':hidden,'layer_2_feats':hidden})
 source=(egcn_h if variant=='H' else egcn_o).EGCN(args,torch.nn.RReLU(),device=device)
 for i,layer in enumerate(source.GRCU_layers):
  layer.GCN_init_weights.data.copy_(saved['model'][f'layers.{i}.initial'])
  for j,gate in enumerate([layer.evolve_weights.update,layer.evolve_weights.reset,layer.evolve_weights.htilda]):
   for name in ['W','U','bias']:getattr(gate,name).data.copy_(saved['model'][f'layers.{i}.gates.{j}.{name}'])
  layer.evolve_weights.choose_topk.scorer.data.copy_(saved['model'][f'layers.{i}.summary.scorer'])
 args=utils.Namespace({'gcn_parameters':{'cls_feats':cls}});head=models.Classifier(args,in_features=hidden*2,out_features=2).to(device);head.load_state_dict(saved['head'])
 errors=[];checked=0;metric_rows=[];logger=original_metric_logger();expected=json.loads((root/'selected.json').read_text());start=time.perf_counter()
 with torch.no_grad():
  for t in range(39,49):
   evidence=np.load(root/f'test-{t}.npz');pairs=evidence['pairs'];rng=torch.load(root/f'test-{t}-rng.pt',map_location='cpu',weights_only=False)
   torch.set_rng_state(rng['cpu'])
   if device.startswith('cuda'):torch.cuda.set_rng_state_all(rng['cuda'])
   elif rng['cuda']:raise ValueError('CUDA random stream needs CUDA replay; CPU is a different experiment')
   # Sparse one-hot features match original data path; weights copied exactly.
   adj=[a.to(device) for a in d['adj'][t-5:t+1]];features=[x.to_sparse().to(device) for x in d['features'][t-5:t+1]];masks=[m.to(device) for m in d['masks'][t-5:t+1]]
   z=source(adj,features,masks);probs=[]
   for start_idx in range(0,len(pairs),100000):
    p=torch.as_tensor(pairs[start_idx:start_idx+100000].astype(np.int64),device=device)
    probs.append(head(torch.cat([z[p[:,0]],z[p[:,1]]],1)).softmax(1)[:,1].cpu().numpy())
   probability=np.concatenate(probs);error=float(np.max(np.abs(probability-evidence['prob'])));errors.append(error);checked+=len(probability)
   assert error<2e-5,(variant,t,error)
   y=evidence['y'];p=evidence['prob'];observed={'map':ap(y,p),'mrr':mrr(pairs,y,p,d['n'])}
   original={'map':float(logger.get_MAP(torch.from_numpy(p),torch.from_numpy(y))), 'mrr':float(logger.get_MRR(torch.from_numpy(p),torch.from_numpy(y),torch.from_numpy(pairs.T.astype(np.int64))))}
   reported=next(row for row in expected['per_snapshot'] if row['t']==t)
   for key in observed:
    assert abs(observed[key]-reported[key])<1e-10,(variant,t,key,'independent',observed[key],reported[key])
    assert abs(original[key]-reported[key])<1e-10,(variant,t,key,'original',original[key],reported[key])
   metric_rows.append({'t':t,'independent':observed,'original':original})
 result={'status':'PASS','metric_rows':metric_rows,'metric_compatibility':'Unedited original metric AST; np.float restored to its historical float alias. Tied-rank ordering checked in pinned NumPy x86 runtime.','variant':variant,'source_predictions_checked':checked,'max_probability_error':max(errors),'per_snapshot_max_error':errors,'seconds':time.perf_counter()-start,'device':device,'scope':'Original EGCN, sparse degree features and original Classifier; copied selected weights; restored per-snapshot stochastic RNG; independent head chunk size','torch':torch.__version__,'verification_environment':{'python':sys.version,'platform':platform.platform(),'gpu':torch.cuda.get_device_name() if device.startswith('cuda') else None,'cuda':torch.version.cuda,'packages':dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata['Name']))}}
 (root/'source_replay.json').write_text(json.dumps(result,indent=2)+'\n');print(result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--variant',required=True);p.add_argument('--device',default='cpu');p.add_argument('--data',default=str(P/'data/l107'));a=p.parse_args();torch.set_num_threads(1);replay(a.root,a.data,a.variant,a.device)
