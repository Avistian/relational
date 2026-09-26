"""Verify notebook CHECK cells reject plausible broken student implementations."""
import ast,json,sys,tempfile
from pathlib import Path
import numpy as np,torch
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'relkit'))
import snapshot_l107 as m
from auth_l107 import authenticate_wiki_cache
root=ast.parse((P/'_build_l107.py').read_text());checks=next(ast.literal_eval(x.value) for x in root.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHECKS' for t in x.targets))
canonical=(P/'relkit/snapshot_l107.py').read_text();tree=ast.parse(canonical)
original={n.name:ast.get_source_segment(canonical,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in checks}
mutations={
'row_mean':('normalized_adjacency',original['normalized_adjacency'].replace('torch.sqrt(degree[i]*degree[j])','degree[i]')),
'drop_multiplicity':('normalized_adjacency',original['normalized_adjacency'].replace('a.values()/torch.sqrt','torch.ones_like(a.values())/torch.sqrt')),
'reversed_gate':('matrix_update',original['matrix_update'].replace('(1-z)*previous+z*proposed','z*previous+(1-z)*proposed')),
'ignored_reset':('matrix_update',original['matrix_update'].replace('candidate(summary,r*previous)','candidate(summary,previous)')),
'strict_close':('completed_bins',original['completed_bins'].replace('<=query','<query')),
'event_only':('completed_bins',original['completed_bins'].replace('(bins+1)*width<=query','np.asarray(times)<query'))}
results={}
for name,(fn,source) in mutations.items():
 ns={'np':np,'torch':torch};exec(source,ns)
 try:exec(checks[fn],ns)
 except (AssertionError,ValueError):results[name]='REJECTED'
 else:raise AssertionError('Mutant survived: '+name)
# Cached raw identity alone must not permit corrupted feature arrays.
authenticate_wiki_cache(P/'data/l102')
with tempfile.TemporaryDirectory() as tmp:
 z=np.load(P/'data/l102/processed.npz');arrays={k:z[k] for k in z.files};arrays['x']=arrays['x'].copy();arrays['x'][0,0]+=1
 np.savez(Path(tmp)/'processed.npz',**arrays)
 try:authenticate_wiki_cache(tmp)
 except ValueError:results['tampered_features']='REJECTED'
 else:raise AssertionError('Tampered raw-feature cache accepted')
r={'status':'PASS','mutants':results};(P/'_mutation_l107_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
