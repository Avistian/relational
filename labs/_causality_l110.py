"""Actual TGN predictions expose tied-memory leakage, and replay is branch-independent."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from relkit.checkpoint_l110 import TGN,temporal_neighbors,batches,strict_batches,capture_checkpoint,restore_checkpoint,evaluate,NegativeSampler
P=Path(__file__).resolve().parent;torch.set_num_threads(1)
u=np.array([1,1,1,2,1,2]);v=np.array([3,4,3,4,3,4]);t=np.array([1.,2.,2.,2.,3.,4.]);e=np.arange(1,7);events={'u':u,'v':v,'t':t,'e':e}
features=np.random.RandomState(41).normal(size=(7,4)).astype('float32');features[0]=0

def trace(batcher,change):
 torch.manual_seed(7);edges=features.copy();edges[2]+=change
 m=TGN(np.zeros((5,4),np.float32),edges,dropout=0,neighbors=2);m.finder=temporal_neighbors(u,v,t,e,5);m.eval();out={}
 for b in batcher(events,2):
  with torch.no_grad():p,_=m.probabilities(b['u'],b['v'],np.full(len(b['u']),4),b['t'],b['e'])
  out.update(zip(b['e'].tolist(),p.tolist()));m.detach_state()
 return out
release,changed=trace(batches,0),trace(batches,50);clean,clean_changed=trace(strict_batches,0),trace(strict_batches,50)
tied_ids=[3,4];gap=max(abs(release[i]-changed[i]) for i in tied_ids);assert gap>1e-7
assert all(clean[i]==clean_changed[i] for i in [1,2,3,4]), 'Equal-time feature influenced a strict-time score'
assert any(clean[i]!=clean_changed[i] for i in [5,6]), 'Intervention never reached later legal predictions'
torch.manual_seed(7);m=TGN(np.zeros((5,4),np.float32),features,dropout=0,neighbors=2);m.finder=temporal_neighbors(u,v,t,e,5)
with torch.no_grad():m.probabilities(u[:1],v[:1],np.array([4]),t[:1],e[:1])
saved=capture_checkpoint(m)
a={k:x[1:] for k,x in events.items()};b={k:x[2:] for k,x in events.items()}
def branches(order):
 out={}
 for name in order:
  restore_checkpoint(m,saved);ev={'a':a,'b':b}[name]
  out[name]=evaluate(m,ev,NegativeSampler(events,2),2,clean=True)
 return out
ab,ba=branches(['a','b']),branches(['b','a'])
for name in ['a','b']:
 assert ab[name][0]==ba[name][0]
 for x,y in zip(ab[name][1],ba[name][1]):
  for key in x:np.testing.assert_array_equal(x[key],y[key])
r={'status':'PASS','release_tied_feature_prediction_gap':gap,'clean_equal_time_noninterference':'EXACT','later_legal_effect':'PRESENT','branch_order_independence':'EXACT','fixture':'Synthetic six-event stream; changes feature at time2, holds weights and candidates fixed'}
(P/'_causality_l110_results.json').write_text(json.dumps(r,indent=2));print(r)
