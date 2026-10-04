"""Execute portable solution with no checkout and no model-download/network calls."""
import json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='b18a-context-state';started=time.monotonic()
n=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
with tempfile.TemporaryDirectory(prefix='b18a-portable-') as td:
    NotebookClient(n,timeout=120,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
nbformat.write(n,P/'solutions'/f'{S}.ipynb');html,_=HTMLExporter().from_notebook_node(n);(P/'html'/f'{S}.html').write_text(html)
r=dict(status='PASS',portable_empty_directory=True,code_cells=sum(c.cell_type=='code' for c in n.cells),seconds=time.monotonic()-started,fresh_notebook_inference='NOT_RUN',saved_inference_replay='PASS',live_colab='NOT_CHECKED')
(P/'_execution_b18a_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
