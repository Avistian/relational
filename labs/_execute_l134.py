"""Execute default portable solution sequentially in an empty working directory."""
import hashlib,json,tempfile,time,os
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MPLBACKEND']='Agg'
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='0134-training-at-scale';path=P/'solutions'/f'{S}.ipynb';nb=nbformat.read(path,as_version=4);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l134-notebook-') as tmp:
 NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 report=json.loads((Path(tmp)/'l134-report.json').read_text());assert report['status']=='PASS' and report['predictions']==6295
 assert report['scale_configurations']==3
 (P/'_teaching_l134_results.json').write_text(json.dumps(report,indent=2))
nbformat.write(nb,path);html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
r=dict(status='PASS',seconds=time.perf_counter()-start,code_cells=sum(c.cell_type=='code' for c in nb.cells),executed_code_sha256=hashlib.sha256('\n\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest(),default_run='Three live scale functions, temporal fixture, three recomputed timing summaries and6295 rescored author predictions; no repo imports',full_training_gate='OFF in default execution; see separate _notebook_gpu_l134_results.json',live_colab='NOT_CHECKED',learner_status='PENDING_WRITTEN_DEFENSE')
(P/'_execution_l134_results.json').write_text(json.dumps(r,indent=2));print(r)
