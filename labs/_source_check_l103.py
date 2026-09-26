"""Compare independently structured port against pinned released model and graph."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from _fetch_l103 import fetch
from relkit.tgat_l103 import TGAT,NeighborFinder
P=Path(__file__).resolve().parent
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
root,_=fetch();source=load('original_tgat',root/'module.py');graph=load('original_graph',root/'graph.py')
device=sys.argv[1] if len(sys.argv)>1 else 'cpu';torch.set_num_threads(1)
rng=np.random.RandomState(7);events={'u':np.array([1,1,2,3,1,2,3,1]),'v':np.array([4,5,4,5,4,5,4,5]),'t':np.array([1.,2.,3.,3.,5.,6.,8.,10.]),'e':np.arange(1,9)}
rows=[[] for _ in range(6)]
for u,v,t,e in zip(events['u'],events['v'],events['t'],events['e']):rows[u].append((v,e,t));rows[v].append((u,e,t))
a=graph.NeighborFinder(rows,uniform=True);b=NeighborFinder(events,6,release=True)
count=0
for node in range(6):
 for t in [0.,1.,2.,3.,4.,6.,11.]:
  np.random.seed(3);x=a.get_temporal_neighbor(np.array([node]),np.array([t]),3)
  np.random.seed(3);y=b.get_temporal_neighbor(np.array([node]),np.array([t]),3)
  for xx,yy in zip(x,y):np.testing.assert_array_equal(xx,yy)
  count+=1
nodes=rng.normal(size=(6,4)).astype('float32');edges=rng.normal(size=(9,4)).astype('float32');nodes[0]=0;edges[0]=0
errors=[];gradient_errors=[]
for training in [False,True]:
 torch.manual_seed(9);ref=source.TGAN(a,nodes,edges,num_layers=2,n_head=2,drop_out=.1).to(device)
 torch.manual_seed(9);own=TGAT(b,nodes,edges).to(device)
 for key,value in ref.state_dict().items():torch.testing.assert_close(value,own.state_dict()[key],rtol=0,atol=0)
 ref.train(training);own.train(training)
 args=(np.array([1,2,3]),np.array([4,5,4]),np.array([5,4,5]),np.array([7.,9.,11.]),3)
 torch.manual_seed(44);np.random.seed(22);rp=ref.contrast(*args);sum(x.sum() for x in rp).backward()
 torch.manual_seed(44);np.random.seed(22);op=own.contrast(*args);sum(x.sum() for x in op).backward()
 for x,y in zip(rp,op):torch.testing.assert_close(x,y,rtol=1e-5,atol=2e-6);errors.append(float((x-y).abs().max()))
 rd=dict(ref.named_parameters());od=dict(own.named_parameters())
 for key,param in rd.items():
  if param.grad is None:assert od[key].grad is None
  else:
   torch.testing.assert_close(param.grad,od[key].grad,rtol=1e-4,atol=3e-6);gradient_errors.append(float((param.grad-od[key].grad).abs().max()))
report={'status':'PASS','device':device,'neighbor_queries':count,'initializer_parity':'EXACT','max_probability_error':max(errors),'max_gradient_error':max(gradient_errors),'modes':['eval','train with dropout'],'historical_environment':'INCOMPARABLE'}
(P/'_source_check_l103_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
