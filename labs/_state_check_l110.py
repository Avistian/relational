"""Failure-oriented tests for warm memory, observation order, and complete checkpoints."""
import json
from pathlib import Path
import numpy as np
import torch
from relkit.checkpoint_l110 import TGN,temporal_neighbors,update_state
P=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.manual_seed(18)
u=np.array([1,1,2,1]);v=np.array([3,4,3,3]);t=np.array([1.,3.,4.,9.]);eid=np.arange(1,5)
model=TGN(np.zeros((5,4),np.float32),np.random.RandomState(3).normal(size=(5,4)).astype(np.float32),dropout=0,neighbors=2)
model.finder=temporal_neighbors(u,v,t,eid,5);model.eval()
model.probabilities(u[:1],v[:1],np.array([4]),t[:1],eid[:1]);model.detach_state()
model.probabilities(u[1:2],v[1:2],np.array([3]),t[1:2],eid[1:2]);model.detach_state()
state=model.snapshot();weights={k:x.clone() for k,x in model.state_dict().items()}
current=(u[2:3],v[2:3],np.array([4]),t[2:3],eid[2:3])
a=model.probabilities(*current);model.restore(state)
# Change only current event feature and a strictly future feature. Neither is a legal input now.
model.edge_features[3]+=10;model.edge_features[4]-=20
b=model.probabilities(*current)
assert all(torch.equal(x,y) for x,y in zip(a,b)), 'Warm-state self/future feature leakage'
# A state_dict-only restoration loses queued messages and therefore does not recover predictions.
model.load_state_dict(weights);model.restore(state)
complete=model.probabilities(*current)
model.load_state_dict(weights);model.restore((state[0],state[1],{}))
incomplete=model.probabilities(*current)
gap=max(float((a-b).abs().max().detach()) for a,b in zip(complete,incomplete))
assert gap>1e-7, 'Counterexample failed to expose the missing pending state'
try:
 update_state(torch.zeros(2,2),torch.tensor([0.,5.]),[1],torch.ones(1,3),torch.tensor([4.]),torch.nn.GRUCell(3,2))
 raise AssertionError('Backward update was accepted')
except AssertionError as ex:
 assert 'backward' in str(ex)
r={'status':'PASS','warm_state_current_and_future_feature_invariance':'EXACT','state_dict_only_missing_pending_prediction_gap':gap,'backward_update_rejected':True}
(P/'_state_check_l110_results.json').write_text(json.dumps(r,indent=2));print(r)
