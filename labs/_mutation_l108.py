"""Show that visible student CHECK cells reject plausible incorrect answers."""
import json
from pathlib import Path
import numpy as np
import nbformat
P=Path(__file__).resolve().parent;nb=nbformat.read(P/'solutions/0108-temporal-neighbor-sampling.ipynb',as_version=4)
checks={}
for i,c in enumerate(nb.cells):
 if c.cell_type=='markdown' and c.source.startswith('### CHECK · '):checks[c.source.split(' · ')[1]]=nb.cells[i+1].source
mutants={
 'inclusive cutoff':('window_bounds',"def window_bounds(t,c,w=np.inf):return int(np.searchsorted(t,c-w,'left')),int(np.searchsorted(t,c,'right'))"),
 'drop singleton':('window_bounds',"def window_bounds(t,c,w=np.inf):return int(np.searchsorted(t,c-w,'left')),max(0,int(np.searchsorted(t,c,'left'))-1)"),
 'repeat recent padding':('choose_positions',"def choose_positions(n,k,p,u):return np.arange(k)%max(n,1)"),
 'forget intermediate levels':('expansion_size',"def expansion_size(l,k):return 1+k**l")}
rejected=[]
for label,(task,code) in mutants.items():
 ns={'np':np};exec(code,ns)
 try:exec(checks[task],ns)
 except AssertionError:rejected.append(label)
 else:raise AssertionError('CHECK failed to reject '+label)
r={'status':'PASS','rejected':rejected,'scope':'Notebook CHECK cells executed against plausible student mutants'}
(P/'_mutation_l108_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
