"""Independent original-source output/gradient and objective parity."""
import sys,json,importlib.util
from pathlib import Path
import numpy as np,torch
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'sources/l107/original'));sys.path.insert(0,str(P/'relkit'))
import snapshot_l107 as m,egcn_h,egcn_o,utils,taskers_utils as tu,Cross_Entropy

def check():
 torch.set_num_threads(1);results={}
 for variant,source in [('H',egcn_h),('O',egcn_o)]:
  args=utils.Namespace({'feats_per_node':5,'layer_1_feats':3,'layer_2_feats':3})
  torch.manual_seed(9);original=source.EGCN(args,torch.nn.RReLU())
  torch.manual_seed(9);port=m.EvolveGCN(5,3,variant)
  for old,new in zip(original.GRCU_layers,port.layers):
   torch.testing.assert_close(old.GCN_init_weights,new.initial)
   for og,ng in zip([old.evolve_weights.update,old.evolve_weights.reset,old.evolve_weights.htilda],new.gates):
    for key in ['W','U','bias']:torch.testing.assert_close(getattr(og,key),getattr(ng,key))
   torch.testing.assert_close(old.evolve_weights.choose_topk.scorer,new.summary.scorer)
  x=[torch.randn(8,5) for _ in range(3)];mask=[torch.zeros(8,1) for _ in x]
  adj=[torch.eye(8).to_sparse() for _ in x]
  torch.manual_seed(17);a=original(adj,x,mask);a.square().sum().backward()
  torch.manual_seed(17);b=port(adj,x,mask);b.square().sum().backward()
  torch.testing.assert_close(a,b,rtol=2e-5,atol=1e-6)
  errors=[]
  for old,new in zip(original.GRCU_layers,port.layers):
   errors.append((old.GCN_init_weights.grad-new.initial.grad).abs().max().item())
   for og,ng in zip([old.evolve_weights.update,old.evolve_weights.reset,old.evolve_weights.htilda],new.gates):
    for key in ['W','U','bias']:
     aa=getattr(og,key).grad;bb=getattr(ng,key).grad;torch.testing.assert_close(aa,bb,rtol=2e-5,atol=1e-6)
   if variant=='H':torch.testing.assert_close(old.evolve_weights.choose_topk.scorer.grad,new.summary.scorer.grad,rtol=2e-5,atol=1e-6)
  results[variant]={'output_max_error':float((a-b).abs().max().detach()),'initial_gradient_max_error':max(errors)}
 logits=torch.randn(19,2);y=torch.arange(19)%2;args=utils.Namespace({'class_weights':[.1,.9],'device':'cpu','task':'link_pred'})
 torch.testing.assert_close(Cross_Entropy.Cross_Entropy(args,None)(logits,y),m.weighted_cross_entropy(logits,y,torch.tensor([.1,.9])))
 return {'status':'PASS','source_commit':'3f4996ac2a742a69fe6ce6e378b6317518bd99bf','models':results,'loss':'PASS'}
if __name__=='__main__':
 r=check();(P/'_source_check_l107_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
