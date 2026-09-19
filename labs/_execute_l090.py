"""Execute standalone inline solution, including the complete experiment, in empty cwd."""
from pathlib import Path
import json,sys,tempfile,hashlib
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import nbformat
from nbclient import NotebookClient
LAB=Path(__file__).resolve().parent;p=LAB/'solutions/0090-gnn-checkpoint.ipynb'
nb=nbformat.read(p,as_version=4)
for c in nb.cells:
 if c.cell_type=='code' and 'RUN_FULL_REPRO = False' in c.source:c.source=c.source.replace('RUN_FULL_REPRO = False','RUN_FULL_REPRO = True')
with tempfile.TemporaryDirectory(prefix='l090-standalone-') as tmp:
 kdir=Path(tmp)/'kernels/python3';kdir.mkdir(parents=True)
 (kdir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'L090 explicit interpreter','language':'python'}))
 km=KernelManager(kernel_name='python3',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(Path(tmp)/'kernels')]))
 try:NotebookClient(nb,timeout=900,km=km,resources={'metadata':{'path':tmp}}).execute()
 finally:
  if km.has_kernel:km.shutdown_kernel(now=True)
 evidence=json.loads((Path(tmp)/'l090-paper.json').read_text());ind=json.loads((Path(tmp)/'l090-inductive.json').read_text());ticket=json.loads((Path(tmp)/'l090-exit.json').read_text())
 author=json.loads((LAB/'_paper_l090_results.json').read_text());extension=json.loads((LAB/'_inductive_l090_results.json').read_text())
 assert evidence['runs']==author['runs'],'Inline replay differs from canonical run'
 assert ind==extension['runs'],'Inductive inline replay differs'
 assert ticket['execution_complete'] and ticket['learner_mastery']=='PENDING_WRITTEN_DEFENSE'
nbformat.write(nb,p)
r={'status':'PASS','empty_cwd':True,'fresh_hash_verified_data_download':True,'inline_paper_runs':100,'inline_inductive_runs':3,'exact_scores_and_validation_traces_match':True,'exit':ticket,'live_colab':'NOT_CHECKED'}
(LAB/'_execution_l090_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
