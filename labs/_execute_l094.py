"""Guarded standalone notebook execution with incremental saved cell progress."""
import argparse,json,os,tempfile,hashlib,resource
from pathlib import Path
# Inherited by the kernel. Cap address space to protect the WSL host.
resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import nbformat
from nbclient import NotebookClient
P=Path(__file__).resolve().parent;path=P/'solutions/0094-hin-survey.ipynb';nb=nbformat.read(path,as_version=4)
ap=argparse.ArgumentParser();ap.add_argument('--full-graph',action='store_true');args=ap.parse_args()
os.environ['L094_RUN_FULL_GRAPH']='1' if args.full_graph else '0';os.environ['L094_DATA_PATH']=str(P/'data/l093/graph_NN.pk')
progress=P/'_execution_l094_progress.json'
def start(cell,cell_index,**kw):
 if cell.cell_type=='code':
  progress.write_text(json.dumps({'status':'RUNNING','cell':cell_index,'prefix':cell.source[:80],'full_graph':args.full_graph})+'\n');print('Executing cell',cell_index,cell.source[:60].splitlines()[0],flush=True)
def done(**kw):nbformat.write(nb,path)
with tempfile.TemporaryDirectory(prefix='l094-inline-') as folder:
 NotebookClient(nb,timeout=120,kernel_name='relational-labs',resources={'metadata':{'path':folder}},on_cell_start=start,on_cell_executed=done).execute()
 if args.full_graph:
  actual=json.loads(Path(folder,'l094-inline-graph.json').read_text());expected=json.loads((P/'_paper_l094_results.json').read_text())['statistics'];assert actual==expected
  (P/'_inline_l094_results.json').write_text(json.dumps(actual,indent=2)+'\n')
 nbformat.write(nb,path)
result={'status':'PASS','scope':'Three live tasks, portable prior-evidence inspection and taxonomy export','code_cells':sum(c.cell_type=='code' for c in nb.cells),'graph_count_parity':'EXACT' if args.full_graph else 'NOT_RUN','fresh_training_runs':0,'source_sha256':hashlib.sha256((P/'relkit/hin_l094.py').read_bytes()).hexdigest(),'live_colab':'NOT_CHECKED','address_space_limit_gib':3,'kernel':'relational-labs','peak_child_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
(P/'_execution_l094_results.json').write_text(json.dumps(result,indent=2)+'\n');progress.write_text(json.dumps({'status':'COMPLETE'})+'\n');print(result)
