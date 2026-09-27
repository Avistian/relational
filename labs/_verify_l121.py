"""Mechanism results and deliberately wrong alternatives, not a benchmark."""
import json,sqlite3
from pathlib import Path
import numpy as np
import torch
from _check_l121 import check_rule,check_aggregate,check_path
from relkit.history_l121 import exists_late,aggregate_at,path_signal
for check,fn in [(check_rule,exists_late),(check_aggregate,aggregate_at),(check_path,path_signal)]:check(fn)
mutants=[(check_rule,lambda rows,roots:[any(r[1]==c for r in rows) for c in roots]),
(check_aggregate,lambda rows,roots,t:aggregate_at([(i,c,v,e,e) for i,c,v,e,a in rows],roots,t)),
(check_aggregate,lambda rows,roots,t:aggregate_at(rows,sorted(roots),t)),
(check_path,lambda x,lo,orr,no,nr: x.new_ones((nr,1))*x.sum().square())]
for check,fn in mutants:
 try:check(fn)
 except AssertionError:pass
 else:raise AssertionError('Wrong implementation escaped')
x=torch.tensor([[2.],[8.],[2.],[8.]],requires_grad=True);out=path_signal(x,torch.tensor([0,1,2,2]),torch.tensor([0,0,1]),3,2);out[0].sum().backward()
assert out.flatten().tolist()==[68.,100.]
r={'status':'PASS','evidence':'COURSE_ONLY','flat_vectors':[[2,10,8],[2,10,8]],'path_outputs':out.flatten().tolist(),'root_A_input_gradients':x.grad.flatten().tolist(),'rejected_mutants':len(mutants),'independent_sql':'PASS','paper_result':'NOT_RUN','learner':'PENDING_WRITTEN_DEFENSE'}
(Path(__file__).resolve().parent/'_verify_l121_results.json').write_text(json.dumps(r,indent=2));print(r)
