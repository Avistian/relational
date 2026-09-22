"""Behavioral contract: strict history, repeated-node aggregation, causal updates."""
from pathlib import Path
import importlib.util
import numpy as np
import torch
P=Path(__file__).resolve().parent
spec=importlib.util.find_spec('relkit.tgn_l102')
assert spec is not None, 'L102 visible implementation has not been created'
from relkit.tgn_l102 import last_message_indices, temporal_neighbors, TGN
assert last_message_indices(np.array([2,1,2,1])).tolist()==[3,2]
u=np.array([1,1,2]);v=np.array([3,4,3]);t=np.array([1.,3.,4.]);e=np.arange(1,4)
f=temporal_neighbors(u,v,t,e,5)
n,ei,ts=f(np.array([1,2,4]),np.array([3.,4.,1.]),2)
assert n.tolist()==[[0,3],[0,0],[0,0]], 'Strict cutoff or padding failed'
model=TGN(np.zeros((5,4),np.float32),np.ones((4,4),np.float32),dropout=0)
model.finder=f
model.eval()
b=(np.array([1]),np.array([3]),np.array([4]),np.array([1.]),np.array([1]))
state=model.snapshot()
p,_=model.probabilities(*b)
model.restore(state)
changed=list(b);changed[4]=np.array([2]);model.edge_features[2]=100
q,_=model.probabilities(*changed)
assert torch.allclose(p,q), 'Current event features leaked into current prediction'
assert len(model.pending)>0 and model.memory.abs().sum()==0, 'First batch must queue rather than self-inform'
model.train();model.detach_state()
p,n=model.probabilities(np.array([1]),np.array([4]),np.array([3]),np.array([3.]),np.array([2]))
(-p.log()-torch.log1p(-n)).sum().backward()
assert model.gru.weight_ih.grad is not None and model.gru.weight_ih.grad.abs().sum()>0, 'Delayed update must train GRU'
print('PASS: strict history, repeated IDs, no self-event leakage, delayed GRU gradient')

import json
(P/'_check_l102_results.json').write_text(json.dumps({'status':'PASS','checks':['last aggregation','strict cutoff and padding','current event feature exclusion','queued first batch','nonzero delayed GRU gradient']},indent=2))
