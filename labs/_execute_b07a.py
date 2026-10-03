"""Execute portable solution from empty directory, using authenticated local checkpoint."""
import hashlib,json,tempfile,time,os
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
P=Path(__file__).resolve().parent;S='b07a-hypernetworks';path=P/'solutions'/(S+'.ipynb');book=nbformat.read(path,4);start=time.monotonic()
os.environ['HYPERFAST_CHECKPOINT']=str(P/'data/b07a/hyperfast.ckpt')
with tempfile.TemporaryDirectory(prefix='b07a-solution-') as td:
 NotebookClient(book,timeout=900,kernel_name='python3',resources={'metadata':{'path':td}}).execute()
 report=json.loads((Path(td)/'b07a-report.json').read_text());assert report['fresh']
 assert report['course']['rows']==json.loads((P/'evidence/b07a/course-audit.json').read_text())['rows']
 assert report['source']==json.loads((P/'evidence/b07a/source-gate.json').read_text())
 assert json.loads((Path(td)/'b07a-submission.json').read_text())['status']=='PENDING_WRITTEN_DEFENSE'
 (P/'evidence/b07a/solution-report.json').write_text(json.dumps(report,indent=2)+'\n')
nbformat.write(book,path)
exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True)
html,_=exporter.from_notebook_node(book);(P/'html'/(S+'.html')).write_text(html)
r=dict(status='PASS',code_cells=sum(c.cell_type=='code' for c in book.cells),working_directory='EMPTY_TEMPORARY',fresh_prediction_metrics='EXACT',seconds=time.monotonic()-start,executed_code_sha256=hashlib.sha256('\n\n'.join(c.source for c in book.cells if c.cell_type=='code').encode()).hexdigest(),live_colab='NOT_CHECKED')
(P/'_execution_b07a_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
