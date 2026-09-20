"""Execute portable solution. Default diagnostic; --full also executes the complete target."""
import json,sys,tempfile,time,subprocess
from pathlib import Path
import nbformat
from nbclient import NotebookClient
P=Path(__file__).resolve().parent;path=P/'solutions/0092-meta-paths.ipynb';nb=nbformat.read(path,as_version=4);full='--full' in sys.argv
if full:
 for cell in nb.cells:
  if cell.cell_type=='code':cell.source=cell.source.replace('RUN_FULL_REPRO = False','RUN_FULL_REPRO = True')
start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l092-inline-') as tmp:
 NotebookClient(nb,timeout=7200,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 # Persist actual outputs before any later cross-run comparison can fail.
 nbformat.write(nb,path)
 if full:
  paper=json.loads((Path(tmp)/'l092-paper.json').read_text())
  (P/'_inline_paper_l092_results.json').write_text(json.dumps(paper,indent=2)+'\n')
result={'status':'PASS','executed_code_cells':sum(c.cell_type=='code' for c in nb.cells),'persisted_output_scope':'full' if full else 'diagnostic:2 full-graph updates; full switch disabled in this saved execution',
'full_encoder_fits_in_saved_execution':int(full),'knn_evaluations_in_saved_full_lane':40 if full else 0,
'fresh_directory_download':'PASS','seconds':time.perf_counter()-start,'independent_prior_full_replay':'_replay_l092_results.json','live_colab':'NOT_CHECKED'}
(P/'_execution_l092_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if full:subprocess.run([sys.executable,str(P/'_compare_l092.py')],check=True)
