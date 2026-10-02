"""Execute every solution cell from an empty directory and require exact full-report parity."""
import hashlib,json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
P=Path(__file__).resolve().parent;S='0183-graph-transformer-pretraining';path=P/'solutions'/(S+'.ipynb');book=nbformat.read(path,4);start=time.monotonic()
with tempfile.TemporaryDirectory(prefix='l183-notebook-') as tmp:
 NotebookClient(book,timeout=180,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 assert json.loads((Path(tmp)/'l183-report.json').read_text())==json.loads((P/'evidence/l183/report.json').read_text())
 assert json.loads((Path(tmp)/'l183-submission.json').read_text())['learner']=='PENDING_WRITTEN_DEFENSE'
nbformat.write(book,path)
exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True)
html,_=exporter.from_notebook_node(book);(P/'html'/(S+'.html')).write_text(html)
r=dict(status='PASS',code_cells=sum(c.cell_type=='code' for c in book.cells),seconds=time.monotonic()-start,working_directory='EMPTY_TEMPORARY',report_parity='EXACT',saved_predictions=7554,raw_label_reconstruction='NOT_RUN',executed_code_sha256=hashlib.sha256('\n\n'.join(c.source for c in book.cells if c.cell_type=='code').encode()).hexdigest(),cloud_usd=0,fresh_training='NOT_RUN',live_colab='NOT_CHECKED',learner='PENDING_WRITTEN_DEFENSE')
(P/'_execution_l183_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
