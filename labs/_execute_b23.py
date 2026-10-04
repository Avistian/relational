"""Execute actual portable solution without a checkout or external data."""
import json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='b23-declared-comparison';start=time.monotonic();n=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
with tempfile.TemporaryDirectory(prefix='b23-portable-') as td:
 NotebookClient(n,timeout=120,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
 assert json.loads((Path(td)/'b23-replay.json').read_text())==json.loads((P/'evidence/b23/report.json').read_text())
nbformat.write(n,P/'solutions'/f'{S}.ipynb');ex=HTMLExporter();ex.register_preprocessor('nbconvert.preprocessors.TagRemovePreprocessor',enabled=True);ex.config.TagRemovePreprocessor.remove_input_tags={'hide-input'}
html,_=ex.from_notebook_node(n);(P/'html'/f'{S}.html').write_text(html)
r=dict(status='PASS',empty_directory=True,exact_report_parity=True,code_cells=sum(c.cell_type=='code' for c in n.cells),seconds=time.monotonic()-start,live_colab='NOT_CHECKED');(P/'_execution_b23_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
