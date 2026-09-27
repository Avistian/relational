"""Meaningful deliberately wrong key/time/query implementations must fail."""
import json
from pathlib import Path
import numpy as np
from _check_l117 import check_edges,check_cutoff,check_targets
from relkit.rdl_l117 import foreign_key_edges,temporal_nodes,query_targets

def collapse_duplicates(pk,fk):
 x=foreign_key_edges(pk,fk);return x[:,:2]
def off_by_one(edges,times,seed,cutoff,hops):return temporal_nodes(edges,times,seed,cutoff-1,hops)
def ignore_hops(edges,times,seed,cutoff,hops):return temporal_nodes(edges,times,seed,cutoff,2)
def include_future(edges,times,seed,cutoff,hops):return temporal_nodes(edges,times,seed,float('inf'),hops)
def sort_queries(targets,ids):return query_targets(targets,np.sort(ids))
def use_prefix(targets,ids):return targets[:len(ids)]
failed=[]
for check,fn in [(check_edges,collapse_duplicates),(check_cutoff,off_by_one),(check_cutoff,ignore_hops),(check_cutoff,include_future),(check_targets,sort_queries),(check_targets,use_prefix)]:
 try:check(fn)
 except AssertionError:failed.append(fn.__name__)
assert len(failed)==6,failed
r={'status':'PASS','rejected_mutations':failed};Path(__file__).with_name('_mutation_l117_results.json').write_text(json.dumps(r,indent=2));print(r)
