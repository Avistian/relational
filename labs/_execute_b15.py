"""Execute portable solution from an empty directory and compare full results."""
import json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='b15-parameter-free-encoders';start=time.monotonic()
n=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
with tempfile.TemporaryDirectory(prefix='b15-portable-') as td:
 NotebookClient(n,timeout=120,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
 assert json.loads((Path(td)/'b15-diagnostic.json').read_text())==json.loads((P/'evidence/b15/diagnostic.json').read_text())
nbformat.write(n,P/'solutions'/f'{S}.ipynb');html,_=HTMLExporter().from_notebook_node(n);(P/'html'/f'{S}.html').write_text(html)
r=dict(status='PASS',portable_empty_directory=True,code_cells=sum(c.cell_type=='code' for c in n.cells),complete_report_equality=True,seconds=time.monotonic()-start)
(P/'_execution_b15_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
