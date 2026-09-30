"""Execute standalone solution in an empty directory with bounded threads."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
P=Path(__file__).resolve().parent;S='0140-rdl-reproduction-checkpoint';path=P/'solutions'/f'{S}.ipynb';notebook=nbformat.read(path,4);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l140-notebook-') as tmp:
 NotebookClient(notebook,timeout=600,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 report=json.loads((Path(tmp)/'l140-report.json').read_text());assert report['status']=='PASS';(P/'_teaching_l140_results.json').write_text(json.dumps(report,indent=2))
nbformat.write(notebook,path);exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True);html,_=exporter.from_notebook_node(notebook);(P/'html'/f'{S}.html').write_text(html)
r=dict(status='PASS',seconds=time.perf_counter()-start,code_cells=sum(c.cell_type=='code' for c in notebook.cells),executed_code_sha256=hashlib.sha256('\n\n'.join(c.source for c in notebook.cells if c.cell_type=='code').encode()).hexdigest(),full_training_gate='NOT_RUN in default notebook',live_colab='NOT_CHECKED',learner='PENDING_WRITTEN_DEFENSE');(P/'_execution_l140_results.json').write_text(json.dumps(r,indent=2));print(r)
