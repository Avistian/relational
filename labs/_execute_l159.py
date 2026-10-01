"""Execute all inline model/trainer cells from an empty directory; compare artifacts."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json,platform,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='0159-foundation-model-preview'
path=P/'solutions'/(S+'.ipynb');notebook=nbformat.read(path,4);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l159-notebook-') as tmp:
    NotebookClient(notebook,timeout=180,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
    result=json.loads((Path(tmp)/'l159-report.json').read_text())
    assert result==json.loads((P/'evidence/l159/mechanism.json').read_text()),'Inline execution differs from canonical module'
    assert json.loads((Path(tmp)/'l159-brief.json').read_text())['learner']=='PENDING_WRITTEN_DEFENSE'
nbformat.write(notebook,path)
html,_=HTMLExporter(template_name='lab').from_notebook_node(notebook);(P/'html'/(S+'.html')).write_text(html)
receipt=dict(status='PASS',seconds=time.perf_counter()-start,code_cells=sum(c.cell_type=='code' for c in notebook.cells),
    working_directory='EMPTY_TEMPORARY',report_parity='EXACT',python=platform.python_version(),
    executed_code_sha256=hashlib.sha256('\n\n'.join(c.source for c in notebook.cells if c.cell_type=='code').encode()).hexdigest(),
    cloud_spend_usd=0,live_colab='NOT_CHECKED',paper_reproduction='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')
(P/'_execution_l159_results.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt)
