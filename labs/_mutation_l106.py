"""Ensure live notebook CHECKs reject meaningful mistakes without downloading data."""
import ast,json
from pathlib import Path
import nbformat
P=Path(__file__).resolve().parent
nb=nbformat.read(P/'solutions/0106-temporal-link-prediction.ipynb',as_version=4)
checks={}
for i,c in enumerate(nb.cells):
 if c.cell_type=='markdown' and c.source.startswith('### CHECK · '):checks[c.source.split(' · ')[1]]=nb.cells[i+1].source
canonical=(P/'relkit/edgebank_l106.py').read_text()
mutants=[('ties admitted','legal_history',canonical.replace('(t<query)','(t<=query)')),('late arrival ignored','legal_history',canonical.replace('(t<query)&(a<=query)','(t<query)')),('window forgotten','memory_scores',canonical.replace("if len(h) and mode=='window':","if False:")),('direction erased','memory_scores',canonical.replace('memory=set(map(tuple,h[:,:2]))','memory={tuple(sorted(x)) for x in h[:,:2]}').replace('tuple(pair) in memory','tuple(sorted(pair)) in memory')),('ties get full credit','binary_metrics',canonical.replace('.5*(tp*fp','.999*(tp*fp'))]
for name,key,source in mutants:
 scope={};exec(compile(source,'<mutant>','exec'),scope)
 try:exec(checks[key],scope)
 except AssertionError:pass
 else:raise AssertionError('Undetected mutant: '+name)
report={'status':'PASS','rejected':[x[0] for x in mutants]}
(P/'_mutation_l106_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
