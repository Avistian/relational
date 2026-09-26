"""Demonstrate that CHECKs reject plausible wrong learner solutions."""
import json
from pathlib import Path
import torch
from _check_l113 import check_mean,check_induced,check_selection
from relkit.scaling_l113 import mean_adjacency,induced_edges,selected_epoch
cases=[('binarize repeated edges',check_mean,lambda e,n:mean_adjacency(torch.unique(e,dim=1),n)),
('transpose sends instead of receives',check_mean,lambda e,n:mean_adjacency(e.flip(0),n)),
('retain global IDs after slicing features',check_induced,lambda e,ids,n:e[:,torch.isin(e[0],ids)&torch.isin(e[1],ids)]),
('test-based checkpoint selection',check_selection,lambda h:max(range(len(h)),key=lambda i:h[i]['test'])),
('last maximum instead of first',check_selection,lambda h:max(range(len(h)),key=lambda i:(h[i]['valid'],i)))]
rows=[]
for name,check,wrong in cases:
 try:check(wrong)
 except (AssertionError,RuntimeError,ValueError):rows.append({'mutation':name,'rejected':True})
 else:raise AssertionError('Escaped mutation: '+name)
r={'status':'PASS','cases':rows};(Path(__file__).parent/'_mutation_l113_results.json').write_text(json.dumps(r,indent=2));print(r)
