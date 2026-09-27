"""Execute the standalone solution in an empty directory; no repo/data dependencies."""
import hashlib,json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='0121-history-relational-ml';path=P/'solutions'/f'{S}.ipynb';nb=nbformat.read(path,as_version=4);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l121-notebook-') as tmp:
 NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 report=json.loads((Path(tmp)/'l121-task-report.json').read_text());assert report['status']=='PASS'
 (P/'_teaching_l121_results.json').write_text(json.dumps(report,indent=2))
nbformat.write(nb,path);html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
r={'status':'PASS','seconds':time.perf_counter()-start,'code_cells':sum(c.cell_type=='code' for c in nb.cells),'executed_code_sha256':hashlib.sha256('\n\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest(),'default_run':'Three live history tasks and separate synthetic GCN smoke run; no repository imports','full_training_gate':'NOT_RUN; real-data timing pilot is separate','live_colab':'NOT_CHECKED','learner_status':'PENDING_WRITTEN_DEFENSE'}
(P/'_execution_l121_results.json').write_text(json.dumps(r,indent=2));print(r)
