"""Execute the portable solution in an empty directory, without repository imports."""
import os,resource
# Inherited by the notebook kernel: fail safely instead of exhausting WSL.
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(300,300))
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='0132-identity-aware-message-passing';path=P/'solutions'/f'{S}.ipynb';nb=nbformat.read(path,as_version=4);start=time.perf_counter()
def progress(cell,cell_index,**kwargs):
 (P/'_execution_l132_progress.json').write_text(json.dumps(dict(cell_index=cell_index,cell_type=cell.cell_type,first_line=cell.source.splitlines()[0][:120] if cell.source else '')))
with tempfile.TemporaryDirectory(prefix='l132-notebook-') as tmp:
 NotebookClient(nb,timeout=180,kernel_name='python3',on_cell_start=progress,resources={'metadata':{'path':tmp}}).execute()
 report=json.loads((Path(tmp)/'l132-report.json').read_text());assert report['status']=='PASS'
 assert report['source_output_max_error']==0 and report['source_gradient_max_error']==0 and report['pilot_rankings_rescored']==8276
 (P/'_teaching_l132_results.json').write_text(json.dumps(report,indent=2))
nbformat.write(nb,path);html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
r=dict(status='PASS',address_space_limit_mib=4096,threads=1,seconds=time.perf_counter()-start,code_cells=sum(c.cell_type=='code' for c in nb.cells),executed_code_sha256=hashlib.sha256('\n\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest(),default_run='Three live identity functions, exact source parity, two synthetic BCE fits and 8276 author pilot rankings; no repo dependencies',full_training_gate='OFF; separate author feasibility/reproduction evidence',live_colab='NOT_CHECKED',learner_status='PENDING_WRITTEN_DEFENSE')
(P/'_execution_l132_results.json').write_text(json.dumps(r,indent=2));print(r)
