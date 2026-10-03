"""Run every portable solution cell from an empty directory and export its results."""
import hashlib,json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
P=Path(__file__).resolve().parent;S='b07-semantic-transfer';path=P/'solutions'/(S+'.ipynb');book=nbformat.read(path,4);start=time.monotonic()
with tempfile.TemporaryDirectory(prefix='b07-solution-') as td:
 NotebookClient(book,timeout=180,kernel_name='python3',resources={'metadata':{'path':td}}).execute()
 assert json.loads((Path(td)/'b07-report.json').read_text())==dict(source=json.loads((P/'evidence/b07/source-gate.json').read_text()),course=json.loads((P/'evidence/b07/course-audit.json').read_text()))
 assert json.loads((Path(td)/'b07-submission.json').read_text())['status']=='PENDING_WRITTEN_DEFENSE'
nbformat.write(book,path)
exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True)
html,_=exporter.from_notebook_node(book);(P/'html'/(S+'.html')).write_text(html)
r=dict(status='PASS',code_cells=sum(c.cell_type=='code' for c in book.cells),working_directory='EMPTY_TEMPORARY',report_parity='EXACT',seconds=time.monotonic()-start,executed_code_sha256=hashlib.sha256('\n\n'.join(c.source for c in book.cells if c.cell_type=='code').encode()).hexdigest(),live_colab='NOT_CHECKED')
(P/'_execution_b07_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
