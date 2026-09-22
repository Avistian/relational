import json,tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='0101-static-vs-temporal'
nb=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
with tempfile.TemporaryDirectory(prefix='l101-standalone-') as d:
 NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':d}}).execute()
nbformat.write(nb,P/'solutions'/f'{S}.ipynb')
html,_=HTMLExporter().from_notebook_node(nb)
(P/'html'/f'{S}.html').write_text(html)
(P/'_execution_l101_results.json').write_text(json.dumps({'status':'PASS','standalone':'Executed in fresh temporary working directory without course imports','code_cells':sum(c.cell_type=='code' for c in nb.cells),'scope':'three task checks, all nine teaching fits and all ten paper target cells','live_colab':'NOT_CHECKED'},indent=2))
print('Standalone solution executed and rendered')
