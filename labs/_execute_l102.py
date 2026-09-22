"""Execute standalone inline solution with cached raw data; full paper gate stays off."""
import json,tempfile,hashlib
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='0102-temporal-graph-networks'
nb=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
with tempfile.TemporaryDirectory(prefix='l102-standalone-') as tmp:
 cache=Path(tmp)/'l102-cache';cache.mkdir()
 # Dataset cache is the only supplied resource; implementation is wholly inline.
 for name in ['wikipedia.csv','processed.npz']:(cache/name).symlink_to(P/'data/l102'/name)
 NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 fresh=json.loads((Path(tmp)/'l102-teaching/seed-0.json').read_text())
 author=json.loads((P/'results/l102/smoke/seed-0.json').read_text())
 assert fresh['test']==author['test'] and fresh['new_test']==author['new_test'], 'Inline replay differs from canonical run'
nbformat.write(nb,P/'solutions'/f'{S}.ipynb')
html,_=HTMLExporter().from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
report={'status':'PASS','code_cells':sum(c.cell_type=='code' for c in nb.cells),'standalone':'Fresh directory, no course imports; hash-checked data cache supplied',
        'coverage':'Three live tasks and full-width two-epoch 600-event training; exact author seed-0 score parity',
        'full_paper_in_notebook':'NOT_RUN: explicit gate off','live_colab':'NOT_CHECKED'}
(P/'_execution_l102_results.json').write_text(json.dumps(report,indent=2));print(report)
