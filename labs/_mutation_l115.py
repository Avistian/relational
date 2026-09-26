"""Student CHECK cells reject plausible incorrect implementations."""
import json
from pathlib import Path
import nbformat
P=Path(__file__).resolve().parent;nb=nbformat.read(P/'solutions/0115-graph-ml-design-patterns.ipynb',as_version=4);ns={}
for cell in nb.cells:
 if cell.cell_type=='code' and not cell.source.startswith(('# RUN:','# Full named')):exec(cell.source,ns)
mutants=[('skip encoder','check_compose',lambda e,p,h,x,a:h(p(x,a),a)),
 ('skip propagating head','check_compose',lambda e,p,h,x,a:p(e(x),a)),
 ('force symmetric directed links','check_pairs',lambda z,p,d=False:ns['pair_features'](z,p,False)),
 ('reverse directed endpoints','check_pairs',lambda z,p,d=False:ns['pair_features'](z,p.flip(0),d)),
 ('sum instead of mean','check_pool',lambda z,b,n:ns['graph_mean'](z,b,n)*ns['torch'].bincount(b,minlength=n).clamp_min(1)[:,None]),
 ('pool across graph IDs','check_pool',lambda z,b,n:z.mean(0).repeat(n,1))]
rows=[]
for name,check,fn in mutants:
 try:ns[check](fn)
 except (AssertionError,ValueError,TypeError,IndexError):rows.append({'mutant':name,'verdict':'REJECTED'})
 else:raise AssertionError('Mutant survived: '+name)
r={'status':'PASS','mutants':rows,'scope':'Actual student CHECK cells, from the generated solution notebook'};(P/'_mutation_l115_results.json').write_text(json.dumps(r,indent=2));print(r)
