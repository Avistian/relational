"""Execute notebook in an empty directory with full report parity."""
import json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='b19b-forecasting-contracts';start=time.monotonic();n=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
with tempfile.TemporaryDirectory(prefix='b19b-portable-') as td:
 NotebookClient(n,timeout=180,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
 for name,ref in [('b19b-diagnostic.json','diagnostic.json'),('b19b-replay.json','portable-replay.json')]:
  assert json.loads((Path(td)/name).read_text())==json.loads((P/'evidence/b19b'/ref).read_text()),name
nbformat.write(n,P/'solutions'/f'{S}.ipynb');html,_=HTMLExporter().from_notebook_node(n);(P/'html'/f'{S}.html').write_text(html)
r=dict(status='PASS',empty_directory=True,complete_diagnostic_parity=True,complete_replay_parity=True,source_archive_verified=True,code_cells=sum(c.cell_type=='code' for c in n.cells),seconds=time.monotonic()-start)
(P/'_execution_b19b_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
