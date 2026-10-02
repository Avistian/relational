"""Execute portable solution from an empty directory, including one fresh full fit."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json,platform,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
P=Path(__file__).resolve().parent;S='0173-multi-task-pretraining';path=P/'solutions'/(S+'.ipynb')
book=nbformat.read(path,4);start=time.monotonic()
with tempfile.TemporaryDirectory(prefix='l173-notebook-') as tmp:
    NotebookClient(book,timeout=180,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
    result=json.loads((Path(tmp)/'l173-fresh-report.json').read_text())
    ref=json.loads((P/'evidence/l173/report.json').read_text())
    expected=next(x for x in ref['runs'] if x['arm']=='cell' and x['seed']==0)
    assert result==expected
    submission=json.loads((Path(tmp)/'l173-submission.json').read_text());assert submission['learner']=='PENDING_WRITTEN_DEFENSE'
nbformat.write(book,path)
exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True)
html,_=exporter.from_notebook_node(book);(P/'html'/(S+'.html')).write_text(html)
r=dict(status='PASS',code_cells=sum(c.cell_type=='code' for c in book.cells),seconds=time.monotonic()-start,working_directory='EMPTY_TEMPORARY',fresh_fit='cell seed0, three complete epochs',fresh_report_parity='EXACT',executed_code_sha256=hashlib.sha256('\n\n'.join(c.source for c in book.cells if c.cell_type=='code').encode()).hexdigest(),live_colab='NOT_CHECKED',cloud_usd=0,learner='PENDING_WRITTEN_DEFENSE')
(P/'_execution_l173_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
