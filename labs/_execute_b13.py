"""Execute portable solution from an empty directory and reconcile both evidence lanes."""
import json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='b13-synthetic-relational-data';start=time.monotonic()
n=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
n.cells.append(nbformat.v4.new_code_cell('print(json.dumps({"report":report,"replay":replay,"verification":verification}))'))
with tempfile.TemporaryDirectory(prefix='b13-empty-') as td:
    NotebookClient(n,timeout=120,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
r=json.loads(''.join(o.get('text','') for o in n.cells[-1].outputs))
assert r['report']==json.loads((P/'evidence/b13/diagnostic.json').read_text())
assert r['replay']==json.loads((P/'evidence/b13/rdbpfn-replay.json').read_text())
n.cells.pop();nbformat.write(n,P/'solutions'/f'{S}.ipynb');html,_=HTMLExporter().from_notebook_node(n);(P/'html'/f'{S}.html').write_text(html)
result=dict(status='PASS',code_cells=sum(c.cell_type=='code' for c in n.cells),working_directory='EMPTY_TEMPORARY',course_exact_report_parity=True,replay_exact_report_parity=True,seconds=time.monotonic()-start,live_colab='NOT_CHECKED')
(P/'_execution_b13_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
