"""Execute the portable solution in an empty directory and check exact report parity."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json,platform,tempfile,time,signal
from importlib.metadata import version
signal.alarm(600)
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
P=Path(__file__).resolve().parent;S='0171-corpus-of-databases';path=P/'solutions'/(S+'.ipynb')
notebook=nbformat.read(path,4);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l171-notebook-') as tmp:
 NotebookClient(notebook,timeout=180,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 result=json.loads((Path(tmp)/'l171-report.json').read_text())
 assert result==json.loads((P/'evidence/l171/report.json').read_text())
 submission=json.loads((Path(tmp)/'l171-submission.json').read_text());assert submission['learner']=='PENDING_WRITTEN_DEFENSE'
nbformat.write(notebook,path)
exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True)
html,_=exporter.from_notebook_node(notebook);(P/'html'/(S+'.html')).write_text(html)
r=dict(status='PASS',code_cells=sum(c.cell_type=='code' for c in notebook.cells),seconds=time.perf_counter()-start,
       python=platform.python_version(),packages={name:version(name) for name in ['nbformat','nbclient','nbconvert','pandas','pyarrow']},working_directory='EMPTY_TEMPORARY',report_parity='EXACT',
       executed_code_sha256=hashlib.sha256('\n\n'.join(c.source for c in notebook.cells if c.cell_type=='code').encode()).hexdigest(),
       new_training='NOT_RUN',additional_cloud_spend_usd=0,live_colab='NOT_CHECKED',learner='PENDING_WRITTEN_DEFENSE')
(P/'_execution_l171_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
