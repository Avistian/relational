"""Run the complete portable solution offline in a clean temporary directory."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json,platform,tempfile,time
from pathlib import Path
import numpy as np
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
P=Path(__file__).resolve().parent;S='0155-compare-manual-fe';path=P/'solutions'/(S+'.ipynb')
notebook=nbformat.read(path,4);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l155-notebook-') as tmp:
 NotebookClient(notebook,timeout=180,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 result=json.loads((Path(tmp)/'l155-report.json').read_text());assert result['status']=='PASS'
 export=json.loads((Path(tmp)/'l155-comparison.json').read_text())
 reference=json.loads((P/'evidence/l155/report.json').read_text())
 assert {k:v for k,v in export.items() if k not in ['written_defense','personal_effort','personal_ratio']}==reference
 (P/'_teaching_l155_results.json').write_text(json.dumps(result,indent=2)+'\n')
nbformat.write(notebook,path)
exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True)
html,_=exporter.from_notebook_node(notebook);(P/'html'/(S+'.html')).write_text(html)
result=dict(status='PASS',code_cells=sum(c.cell_type=='code' for c in notebook.cells),
            seconds=time.perf_counter()-start,python=platform.python_version(),numpy=np.__version__,
            nbformat=nbformat.__version__,
            executed_code_sha256=hashlib.sha256('\n\n'.join(c.source for c in notebook.cells if c.cell_type=='code').encode()).hexdigest(),
            working_directory='EMPTY_TEMPORARY',report_parity='EXACT',fresh_training='NOT_RUN',
            additional_cloud_spend_usd=0,live_colab='NOT_CHECKED',learner='PENDING_WRITTEN_DEFENSE')
(P/'_execution_l155_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
