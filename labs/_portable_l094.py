"""Replay all solution code under the pinned portable dependencies, without repo imports."""
import os,resource,json,tempfile,hashlib,importlib.metadata as md
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',L094_RUN_FULL_GRAPH='1')
P=Path(__file__).resolve().parent;os.environ['L094_DATA_PATH']=str(P/'data/l093/graph_NN.pk')
nb=json.loads((P/'solutions/0094-hin-survey.ipynb').read_text());namespace={};cells=[c for c in nb['cells'] if c['cell_type']=='code']
with tempfile.TemporaryDirectory(prefix='l094-portable-') as tmp:
 os.chdir(tmp)
 for i,c in enumerate(cells):
  print('Portable cell',i,flush=True);exec(compile(''.join(c['source']),f'<cell-{i}>','exec'),namespace)
 observed=json.loads(Path('l094-inline-graph.json').read_text());assert observed==json.loads((P/'_paper_l094_results.json').read_text())['statistics']
result={'status':'PASS','environment':{x:md.version(x) for x in ['numpy','pandas','dill']},'code_cells':len(cells),'full_NN_recount':'EXACT','peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'address_space_limit_gib':3,'live_colab':'NOT_CHECKED','scope':'Fresh portable interpreter executing all solution code; no notebook frontend claim'}
(P/'_portable_l094_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
