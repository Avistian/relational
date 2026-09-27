"""Reject plausible semantic errors, rather than testing only expected examples."""
import json,torch
from pathlib import Path
from _check_l132 import check_marker,check_targets,check_map
from relkit import identity_l132 as m
mutants=[('all_nodes_marked',check_marker,lambda x,b,e:x+e),('detached_marker',check_marker,lambda x,b,e:m.mark_roots(x,b,e.detach())),('global_id_membership',check_targets,lambda owner,ids,po,pi,b:torch.isin(ids,pi).float()),('all_positives',check_targets,lambda owner,ids,po,pi,b:torch.ones(len(ids))),('pooled_precision',check_map,lambda p,t,k:sum(sum(x in y for x in a) for a,y in zip(p,t))/(len(p)*k)),('constant_metric',check_map,lambda p,t,k:2/3)]
r=[]
for name,check,fn in mutants:
 try:check(fn)
 except Exception as e:r.append(dict(mutant=name,status='REJECTED',exception=type(e).__name__))
 else:raise AssertionError(name+' survived')
Path(__file__).with_name('_mutation_l132_results.json').write_text(json.dumps(dict(status='PASS',mutants=r),indent=2));print(r)
