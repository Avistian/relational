"""Execute the complete inline solution in an isolated working directory."""
import hashlib,json,os,tempfile
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import nbformat
from nbclient import NotebookClient
P=Path(__file__).resolve().parent;path=P/'solutions/0095-bipartite-graphs.ipynb';nb=nbformat.read(path,as_version=4)
os.environ['L095_DATA_PATH']=str(P/'data/l095/ml-100k.zip')
with tempfile.TemporaryDirectory(prefix='l095-inline-') as folder:
    NotebookClient(nb,timeout=180,kernel_name='relational-labs',resources={'metadata':{'path':folder}}).execute()
    actual=json.loads(Path(folder,'l095-fresh.json').read_text());expected=json.loads((P/'_experiment_l095_results.json').read_text())
    assert actual['audit']==expected['release_audit']
    assert actual['experiment']['summary']==expected['summary']
    assert actual['experiment']['runs']==expected['runs']
    (P/'_inline_l095_results.json').write_text(json.dumps(actual,indent=2)+'\n')
nbformat.write(nb,path)
result={'status':'PASS','code_cells':sum(c.cell_type=='code' for c in nb.cells),'fresh_folds':5,'full_runner_inline_parity':'EXACT','source_sha256':hashlib.sha256((P/'relkit/bipartite_l095.py').read_bytes()).hexdigest(),'live_colab':'NOT_CHECKED'}
(P/'_execution_l095_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
