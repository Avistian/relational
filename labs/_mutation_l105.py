"""Prove notebook CHECK cells reject plausible errors in each live function."""
import contextlib,io,json
from pathlib import Path
import nbformat
P=Path(__file__).resolve().parent
nb=nbformat.read(P/'solutions/0105-continuous-time.ipynb',as_version=4)
cells=[c.source for c in nb.cells if c.cell_type=='code']
mutants={
 'right_boundary_in_previous_bin':('return bins.astype(np.int64)','return np.ceil((t-origin)/width).astype(np.int64)-1'),
 'binary_counts_lose_multiplicity':("'count':counts,","'count':np.ones_like(counts),"),
 'release_ignores_late_arrival':("'release':np.maximum(end,window_arrival)","'release':end"),
 'strict_release_excludes_completed_boundary':("snapshots['release']<=query","snapshots['release']<query")}
results={}
for name,(before,after) in mutants.items():
 env={};changed=False;rejected=False
 try:
  with contextlib.redirect_stdout(io.StringIO()):
   for code in cells[1:]:
    if code.startswith('# PROVIDED — complete raw-data replay.'):break
    if before in code:code=code.replace(before,after);changed=True
    exec(compile(code,'notebook CHECK mutation '+name,'exec'),env)
 except AssertionError:rejected=True
 assert changed and rejected,name
 results[name]='REJECTED_BY_NOTEBOOK_CHECK'
(P/'_mutation_l105_results.json').write_text(json.dumps({'status':'PASS','mutants':results},indent=2)+'\n');print(results)
