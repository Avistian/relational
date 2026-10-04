"""Execute default solution from an empty directory; compare complete reports."""
import json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='b20-curriculum-order';start=time.monotonic();n=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
with tempfile.TemporaryDirectory(prefix='b20-portable-') as td:
 NotebookClient(n,timeout=120,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
 assert json.loads((Path(td)/'b20-replay.json').read_text())==json.loads((P/'_verify_b20_results.json').read_text())['scores']
 assert json.loads((Path(td)/'b20-paper-replay.json').read_text())==json.loads((P/'evidence/b20/reproduction.json').read_text())
nbformat.write(n,P/'solutions'/f'{S}.ipynb');exporter=HTMLExporter();exporter.register_preprocessor('nbconvert.preprocessors.TagRemovePreprocessor',enabled=True);exporter.config.TagRemovePreprocessor.remove_input_tags={'hide-input'};html,_=exporter.from_notebook_node(n);(P/'html'/f'{S}.html').write_text(html)
out=dict(status='PASS',empty_directory=True,complete_course_replay=True,complete_printed_table_replay=True,code_cells=sum(c.cell_type=='code' for c in n.cells),seconds=time.monotonic()-start)
(P/'_execution_b20_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
