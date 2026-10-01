"""Execute the portable default audit in an empty working directory."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json,platform,tempfile,time
from pathlib import Path
import numpy as np,nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
P=Path(__file__).resolve().parent;S='0156-temporal-leakage-audit';path=P/'solutions'/(S+'.ipynb');notebook=nbformat.read(path,4);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l156-execute-') as tmp:
 NotebookClient(notebook,timeout=180,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 result=json.loads((Path(tmp)/'l156-report.json').read_text());assert result['status']=='PASS'
 report=json.loads((Path(tmp)/'l156-audit-report.json').read_text());report.pop('written_defense')
 assert report==json.loads((P/'evidence/l156/report.json').read_text())
nbformat.write(notebook,path)
exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True)
html,_=exporter.from_notebook_node(notebook);(P/'html'/(S+'.html')).write_text(html)
r=dict(status='PASS',code_cells=sum(c.cell_type=='code' for c in notebook.cells),seconds=time.perf_counter()-start,python=platform.python_version(),numpy=np.__version__,working_directory='EMPTY_TEMPORARY',report_parity='EXACT',executed_code_sha256=hashlib.sha256('\n\n'.join(c.source for c in notebook.cells if c.cell_type=='code').encode()).hexdigest(),full_gate='OFF',live_colab='NOT_CHECKED',learner='PENDING_WRITTEN_DEFENSE')
(P/'_execution_l156_results.json').write_text(json.dumps(r,indent=2));print(r)
