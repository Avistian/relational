"""Execute complete notebook offline in an empty directory; compare both lanes."""
import json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='b19a-predictive-distributions';start=time.monotonic();n=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
with tempfile.TemporaryDirectory(prefix='b19a-portable-') as td:
 NotebookClient(n,timeout=180,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
 assert json.loads((Path(td)/'b19a-diagnostic.json').read_text())==json.loads((P/'evidence/b19a/diagnostic.json').read_text())
 assert json.loads((Path(td)/'b19a-replay.json').read_text())==json.loads((P/'evidence/b19a/portable-replay.json').read_text())
 effects=json.loads((Path(td)/'b19a-paired-effects.json').read_text());assert len(effects)==97
 (P/'evidence/b19a/paired-effects.json').write_text(json.dumps(effects,indent=2)+'\n')
nbformat.write(n,P/'solutions'/f'{S}.ipynb');html,_=HTMLExporter().from_notebook_node(n);(P/'html'/f'{S}.html').write_text(html)
r=dict(status='PASS',empty_directory=True,complete_diagnostic_parity=True,complete_replay_parity=True,paired_dataset_effects=len(effects),code_cells=sum(c.cell_type=='code' for c in n.cells),seconds=time.monotonic()-start)
(P/'_execution_b19a_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
