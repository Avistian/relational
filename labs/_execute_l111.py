"""Execute the visible default notebook and preserve its outputs and code identity."""
import hashlib,json,os,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0111-ogb-benchmark-contract'
os.environ['L111_ARCHIVE']=str(P/'data/l112/arxiv.zip')
p=P/'solutions'/f'{S}.ipynb';book=nbformat.read(p,as_version=4);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l111-notebook-') as tmp:
 NotebookClient(book,timeout=240,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 report=json.loads((Path(tmp)/'l111-report.json').read_text())
 assert report==json.loads((P/'evidence/l111/summary.json').read_text())
nbformat.write(book,p);html,_=HTMLExporter(template_name='lab').from_notebook_node(book);(P/'html'/f'{S}.html').write_text(html)
record=dict(status='PASS',seconds=time.perf_counter()-start,code_sha256=hashlib.sha256('\n\n'.join(c.source for c in book.cells if c.cell_type=='code').encode()).hexdigest(),scope='Full official OGB loader, official/independent evaluator agreement, two live functions; no GCN fit',live_colab='NOT_CHECKED')
(P/'_execution_l111_results.json').write_text(json.dumps(record,indent=2)+'\n');print(record)
