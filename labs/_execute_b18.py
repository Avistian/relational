"""Execute self-contained solution from an empty directory; compare complete evidence."""
import gzip,json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='b18-context-sufficiency';start=time.monotonic()
n=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
with tempfile.TemporaryDirectory(prefix='b18-portable-') as td:
    NotebookClient(n,timeout=120,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
    assert json.loads(gzip.decompress((Path(td)/'b18-diagnostic.json.gz').read_bytes()))==json.loads(gzip.decompress((P/'evidence/b18/diagnostic.json.gz').read_bytes()))
nbformat.write(n,P/'solutions'/f'{S}.ipynb');html,_=HTMLExporter().from_notebook_node(n);(P/'html'/f'{S}.html').write_text(html)
r=dict(status='PASS',portable_empty_directory=True,complete_report_equality=True,code_cells=sum(c.cell_type=='code' for c in n.cells),seconds=time.monotonic()-start)
(P/'_execution_b18_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
