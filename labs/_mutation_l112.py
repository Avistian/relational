"""Reject plausible wrong answers using the actual notebook CHECK functions."""
import json
from pathlib import Path
import nbformat
import torch
from relkit.ogb_l112 import normalized_adjacency,training_loss,selected_epoch
P=Path(__file__).resolve().parent;nb=nbformat.read(P/'solutions/0112-ogb-gcn-reproduction.ipynb',as_version=4)
namespace={'torch':torch,'normalized_adjacency':normalized_adjacency,'training_loss':training_loss,'selected_epoch':selected_epoch}
for cell in nb.cells:
 if cell.cell_type=='code' and cell.source.startswith('def check_'):exec(cell.source,namespace)
mutants=[('missing_self_loops','check_normalization',lambda e,n:torch.zeros(n,n).to_sparse()),('all_labels_loss','check_loss',lambda p,y,i:torch.nn.functional.nll_loss(p,y)),('last_epoch','check_selection',lambda v:len(v)-1),('last_tied_maximum','check_selection',lambda v:len(v)-1-list(v)[::-1].index(max(v)))]
rejected=[]
for name,check,fn in mutants:
 try:namespace[check](fn)
 except (AssertionError,ValueError):rejected.append(name)
 else:raise AssertionError('Surviving mutant: '+name)
r={'status':'PASS','rejected':rejected,'checks_source':'Executed solution notebook cells','learner_status':'PENDING_WRITTEN_DEFENSE'};(P/'_mutation_l112_results.json').write_text(json.dumps(r,indent=2));print(r)
