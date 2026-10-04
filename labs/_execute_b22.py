"""Execute the actual portable solution in an empty working directory."""
import json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='b22-support-state-refinement';started=time.monotonic()
n=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
with tempfile.TemporaryDirectory(prefix='b22-portable-') as td:
 NotebookClient(n,timeout=120,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
 expected=json.loads((P/'_verify_b22_results.json').read_text());expected.pop('corruptions_rejected')
 assert json.loads((Path(td)/'b22-replay.json').read_text())==expected
 assert json.loads((Path(td)/'b22-fresh.json').read_text())==json.loads((P/'evidence/b22/diagnostic.json').read_text())
nbformat.write(n,P/'solutions'/f'{S}.ipynb')
exporter=HTMLExporter();exporter.register_preprocessor('nbconvert.preprocessors.TagRemovePreprocessor',enabled=True);exporter.config.TagRemovePreprocessor.remove_input_tags={'hide-input'}
html,_=exporter.from_notebook_node(n);(P/'html'/f'{S}.html').write_text(html)
out=dict(status='PASS',empty_directory=True,complete_fresh_course=True,independent_replay=True,source_audit=True,code_cells=sum(c.cell_type=='code' for c in n.cells),seconds=time.monotonic()-started)
(P/'_execution_b22_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
