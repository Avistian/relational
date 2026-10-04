"""Gradient, time, reverse-edge and coupled-edit falsification checks."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from relkit.structural_b21 import fixture,parameters,forward,adjacency,legal_states,direction_score,validate_assignment
P=Path(__file__).resolve().parent;d=fixture();a=adjacency(d['original']);rows=[];eps=1e-6
for seed in range(3):
    w=parameters(seed);f=torch.tensor(a,requires_grad=True);r=torch.tensor(a.T,requires_grad=True);y=torch.tensor(d['target'])
    def loss(af,ar):return ((forward(d,w,af,ar)-y)**2).mean()
    clean=loss(f,r);clean.backward()
    for state in legal_states(d,2):
        if sum(state[i]!=d['original'][i] for i in range(6))==0:continue
        # One changed session, possibly two changed FK cells.
        sessions={d['session'][i] for i in range(6) if state[i]!=d['original'][i]}
        if len(sessions)!=1:continue
        delta=adjacency(state)-a
        analytic=direction_score(f.grad.numpy(),r.grad.numpy(),a,a+delta)
        # One-sided derivative stays on nonnegative edge weights.
        h=float((loss(torch.tensor(a+eps*delta),torch.tensor((a+eps*delta).T))-clean.detach())/eps)
        rows.append(dict(seed=seed,state=state,analytic=analytic,finite_difference=h,error=abs(h-analytic)))
    # Isolated future parent cannot affect eligible predictions.
    future=copy.deepcopy(d);future['parent_x'][3]=[1e6,-1e6]
    np.testing.assert_array_equal(forward(d,w,torch.tensor(a),torch.tensor(a.T)).detach().numpy(),forward(future,w,torch.tensor(a),torch.tensor(a.T)).detach().numpy())
    hidden=copy.deepcopy(d);hidden['target']=[99,-99,123]
    np.testing.assert_array_equal(forward(d,w,torch.tensor(a),torch.tensor(a.T)).detach().numpy(),forward(hidden,w,torch.tensor(a),torch.tensor(a.T)).detach().numpy())
assert len(rows)==30 and max(x['error'] for x in rows)<1e-5
out=dict(status='PASS',directions=rows,max_direction_error=max(x['error'] for x in rows),future_feature_interventions=3,query_label_exclusion_interventions=3)
(P/'evidence/b21/mechanism-checks.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='directions'})
