"""Actual student CHECK cells must reject incorrect live task implementations."""
import json
from pathlib import Path
import nbformat
import numpy as np
P=Path(__file__).resolve().parent;nb=nbformat.read(P/'solutions/0114-ogb-error-analysis.ipynb',as_version=4);ns={}
for cell in nb.cells:
 if cell.cell_type=='code' and not cell.source.startswith(('# RUN:','# Full named')):exec(cell.source,ns)
mutants=[('isolates become zero','check_neighborhood',lambda e,y,t:{k:np.nan_to_num(v) for k,v in ns['neighborhood_properties'](e,y,t).items()}),
 ('homophily becomes training coverage','check_neighborhood',lambda e,y,t:{**ns['neighborhood_properties'](e,y,t),'homophily':ns['neighborhood_properties'](e,y,t)['train_neighbor_fraction']}),
 ('ignores selected IDs','check_metrics',lambda y,g,m,i:ns['slice_metrics'](y,g,m,np.arange(len(y)))),
 ('drops support filter','check_choice',lambda rows,minimum_count=200:min(rows,key=lambda r:r['mean_delta_pp']))]
rows=[]
for name,check,fn in mutants:
 try:ns[check](fn)
 except (AssertionError,ValueError,TypeError,IndexError):rows.append({'mutant':name,'verdict':'REJECTED'})
 else:raise AssertionError('Mutant survived: '+name)
r={'status':'PASS','mutants':rows,'scope':'Student CHECK source executed from solution notebook; no import of canonical task functions'};(P/'_mutation_l114_results.json').write_text(json.dumps(r,indent=2));print(r)
