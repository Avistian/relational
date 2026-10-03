"""Execute self-contained solution in an empty directory; require report parity."""
import hashlib,json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='b12-adaptation-mechanisms';start=time.monotonic()
n=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
n.cells.append(nbformat.v4.new_code_cell('print(json.dumps({"report":report,"verification":verification}))'))
with tempfile.TemporaryDirectory(prefix='b12-notebook-') as td:NotebookClient(n,timeout=90,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
r=json.loads(''.join(o.get('text','') for o in n.cells[-1].outputs))
assert r['report']==json.loads((P/'evidence/b12/diagnostic.json').read_text())
assert r['verification']==json.loads((P/'_verify_b12_results.json').read_text())
n.cells.pop();nbformat.write(n,P/'solutions'/f'{S}.ipynb');html,_=HTMLExporter().from_notebook_node(n);(P/'html'/f'{S}.html').write_text(html)
result=dict(status='PASS',code_cells=sum(c.cell_type=='code' for c in n.cells),working_directory='EMPTY_TEMPORARY',exact_author_report_parity=True,seconds=time.monotonic()-start,live_colab='NOT_CHECKED',code_sha256=hashlib.sha256('\n'.join(c.source for c in n.cells if c.cell_type=='code').encode()).hexdigest())
(P/'_execution_b12_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
