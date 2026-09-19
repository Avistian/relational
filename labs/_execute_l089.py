"""Execute full inline solution in an empty working directory; smoke only, not paper."""
from pathlib import Path
import hashlib,json,tempfile,sys
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import nbformat
from nbclient import NotebookClient
LAB=Path(__file__).resolve().parent;p=LAB/'solutions/0089-sampling-at-scale.ipynb'
nb=nbformat.read(p,as_version=4)
with tempfile.TemporaryDirectory(prefix='l089-notebook-') as tmp:
 kdir=Path(tmp)/'kernels/python3';kdir.mkdir(parents=True)
 (kdir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'L089 explicit interpreter','language':'python'}))
 km=KernelManager(kernel_name='python3',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(Path(tmp)/'kernels')]))
 try:
  NotebookClient(nb,timeout=600,km=km,resources={'metadata':{'path':tmp}}).execute()
 finally:
  if km.has_kernel: km.shutdown_kernel(now=True)
 exit_artifact=json.loads((Path(tmp)/'l089-exit.json').read_text());assert not exit_artifact['complete']
 assert not (Path(tmp)/'l089-paper').exists()
 sha=hashlib.sha256((Path(tmp)/'l089-data/ppi.zip').read_bytes()).hexdigest()
nbformat.write(nb,p)
r={'status':'PASS','empty_cwd':True,'fresh_download_sha256':sha,'code_cells':sum(c.cell_type=='code' for c in nb.cells),'inline_complete_model_and_trainer':True,'smoke':exit_artifact['smoke'],'exit_complete':False,'full_paper_in_notebook':'NOT_RUN','live_colab':'NOT_CHECKED'}
(LAB/'_execution_l089_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
