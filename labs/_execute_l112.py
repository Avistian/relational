"""Run the solution with a fresh raw-data work directory and live task code."""
import hashlib,json,os,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='0112-ogb-gcn-reproduction';path=P/'solutions'/f'{S}.ipynb';nb=nbformat.read(path,as_version=4);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l112-notebook-') as tmp:
 d=Path(tmp)/'l112-data';d.mkdir();os.symlink(P/'data/l112/arxiv.zip',d/'arxiv.zip')
 NotebookClient(nb,timeout=900,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 fresh=json.loads((Path(tmp)/'l112-fresh.json').read_text())
nbformat.write(nb,path);html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
r={'status':'PASS','seconds':time.perf_counter()-start,'fresh':fresh,'code_cells':sum(c.cell_type=='code' for c in nb.cells),'executed_code_sha256':hashlib.sha256('\n\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest(),'full_training_gate':'OFF; separate GPU author evidence','live_colab':'NOT_CHECKED','learner_status':'PENDING_WRITTEN_DEFENSE'}
(P/'_execution_l112_results.json').write_text(json.dumps(r,indent=2));print(r)
