"""Does changing an earlier snapshot actually change the final prediction?"""
import sys,json
from pathlib import Path
import torch
from unittest.mock import patch
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'relkit'))
from snapshot_l107 import EvolveGCN
r={}
for variant in ['H','O']:
 torch.manual_seed(22);model=EvolveGCN(4,3,variant);a=[torch.eye(7)]*6;x=[torch.randn(7,4) for _ in range(6)];m=[torch.zeros(7,1)]*6
 changed=[v.clone() for v in x];changed[0]=changed[0]*7+5
 with patch('torch.nn.functional.rrelu',side_effect=lambda x,**kw:torch.nn.functional.leaky_relu(x,11/48)):
  y=model(a,x,m);z=model(a,changed,m)
 gap=float((y-z).abs().max().detach());r[variant]=gap
assert r['H']>1e-7 and r['O']==0,r
out={'status':'PASS','intervention':'First of six feature matrices changed; final graph/features, model weights; RReLU replaced by its fixed mean slope (a separate deterministic mechanism audit)','max_output_change':r,'interpretation':'Released O resets weight recurrence for each six-slice query and never conditions it on observations; earlier graph content cannot affect the final output. This statement concerns the released fixed-length window protocol.'}
(P/'_history_witness_l107_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
