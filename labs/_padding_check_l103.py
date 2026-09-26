"""A source-confirmed counterexample to full-model clock-shift invariance."""
import importlib.util,json
from pathlib import Path
import numpy as np,torch
from _fetch_l103 import fetch
import relkit.tgat_l103 as own
P=Path(__file__).resolve().parent;root,_=fetch();torch.set_num_threads(1)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
source=load('original_tgat',root/'module.py');graph=load('original_graph',root/'graph.py')
events={'u':np.array([],dtype=int),'v':np.array([],dtype=int),'e':np.array([],dtype=int),'t':np.array([],dtype=float)}
nodes=np.zeros((5,4),np.float32);edges=np.zeros((1,4),np.float32)
torch.manual_seed(3);ref=source.TGAN(graph.NeighborFinder([[] for _ in range(5)],uniform=True),nodes,edges,num_layers=2,n_head=2,drop_out=0).eval()
torch.manual_seed(3);port=own.TGAT(own.NeighborFinder(events,5),nodes,edges,layers=2,dropout=0).eval()
u=np.array([1,2]);v=np.array([3,4]);neg=np.array([4,3]);t=np.array([2.,3.]);reported=[]
with torch.no_grad():
 for shift in [0.,10.]:
  expected=ref.contrast(u,v,neg,t+shift,3)[0];got=port.contrast(u,v,neg,t+shift,3)[0]
  torch.testing.assert_close(expected,got,rtol=0,atol=0);reported.append(got.numpy().tolist())
assert not np.allclose(reported[0],reported[1]),'Expected release padding to expose the fixed time-zero origin'
original=own.attention_weights
try:
 def zero_safe(q,k,padding):return original(q,k,padding).masked_fill(padding.all(dim=-1,keepdim=True),0)
 own.attention_weights=zero_safe
 with torch.no_grad():
  a=port.contrast(u,v,neg,t,3)[0];b=port.contrast(u,v,neg,t+10,3)[0];torch.testing.assert_close(a,b,rtol=0,atol=0)
finally:own.attention_weights=original
report={'status':'PASS','scope':'Synthetic no-history intervention, same weights and node/edge inputs; not a trained result','original_model_parity':'EXACT','query_times':[2.,3.],'shifted_times':[12.,13.],'released_probabilities':reported,'max_release_shift_difference':float(np.max(np.abs(np.array(reported[0])-reported[1]))),'zero_safe_probabilities':a.tolist(),'zero_safe_shift_difference':float((a-b).abs().max()),'conclusion':'Real-event elapsed inputs are shift invariant; fixed zero-time padding plus uniform all-padding attention breaks that property for the complete released model.'}
(P/'_padding_check_l103_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
