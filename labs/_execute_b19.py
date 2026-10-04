"""Execute both evidence lanes from an empty directory, compare all outputs."""
import json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from _reproduce_b19 import replay_grouped
P=Path(__file__).resolve().parent;S='b19-benchmark-evidence';start=time.monotonic()
n=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
with tempfile.TemporaryDirectory(prefix='b19-portable-') as td:
    NotebookClient(n,timeout=120,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
    assert json.loads((Path(td)/'b19-diagnostic.json').read_text())==json.loads((P/'evidence/b19/diagnostic.json').read_text())
    expected=replay_grouped(json.loads((P/'evidence/b19/released-grouped-scores.json').read_text()))
    assert json.loads((Path(td)/'b19-grouped-summary.json').read_text())==expected
nbformat.write(n,P/'solutions'/f'{S}.ipynb');html,_=HTMLExporter().from_notebook_node(n);(P/'html'/f'{S}.html').write_text(html)
r=dict(status='PASS',portable_empty_directory=True,complete_diagnostic_equality=True,complete_grouped_summary_equality=True,code_cells=sum(c.cell_type=='code' for c in n.cells),seconds=time.monotonic()-start)
(P/'_execution_b19_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
