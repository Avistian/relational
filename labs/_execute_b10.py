"""Execute portable solution in an empty directory; export the executed notebook."""
import hashlib,json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='b10-relational-transformer';start=time.monotonic()
n=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
with tempfile.TemporaryDirectory(prefix='b10-notebook-') as td:NotebookClient(n,timeout=90,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
nbformat.write(n,P/'solutions'/f'{S}.ipynb');html,_=HTMLExporter().from_notebook_node(n);(P/'html'/f'{S}.html').write_text(html)
r=dict(status='PASS',code_cells=sum(c.cell_type=='code' for c in n.cells),working_directory='EMPTY_TEMPORARY',mode='CPU_MECHANISM_AND_FULL_SAVED_CONTEXT_REPLAY',seconds=time.monotonic()-start,code_sha256=hashlib.sha256('\n'.join(c.source for c in n.cells if c.cell_type=='code').encode()).hexdigest(),live_colab='NOT_CHECKED')
(P/'_execution_b10_results.json').write_text(json.dumps(r,indent=2));print(r)
