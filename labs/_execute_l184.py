"""Execute solution from an empty directory and compare the complete report."""
import hashlib,json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
P=Path(__file__).resolve().parent;S='0184-gelgt-temporal-attention';path=P/'solutions'/(S+'.ipynb');book=nbformat.read(path,4);start=time.monotonic()
with tempfile.TemporaryDirectory(prefix='l184-notebook-') as tmp:
 NotebookClient(book,timeout=180,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 assert json.loads((Path(tmp)/'l184-report.json').read_text())==json.loads((P/'evidence/l184/report.json').read_text())
nbformat.write(book,path)
exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True)
html,_=exporter.from_notebook_node(book);(P/'html'/(S+'.html')).write_text(html)
r=dict(status='PASS',code_cells=sum(c.cell_type=='code' for c in book.cells),seconds=time.monotonic()-start,working_directory='EMPTY_TEMPORARY',report_parity='EXACT',executed_code_sha256=hashlib.sha256('\n\n'.join(c.source for c in book.cells if c.cell_type=='code').encode()).hexdigest(),cloud_usd=0,live_colab='NOT_CHECKED')
(P/'_execution_l184_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
