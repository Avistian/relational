"""Run the portable notebook's full evaluation lane using a copied pinned cache."""
import hashlib,json,os,shutil,tempfile,time
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MPLBACKEND']='Agg'
import nbformat
from nbclient import NotebookClient
P=Path(__file__).resolve().parent
nb=nbformat.read(P/'solutions/0136-leaderboard-literacy.ipynb',4)
code='\n\n'.join(c.source for c in nb.cells if c.cell_type=='code');digest=hashlib.sha256(code.encode()).hexdigest()
for c in nb.cells:
 if c.cell_type=='code':c.source=c.source.replace('RUN_FULL_LEADERBOARD_REPLAY = False','RUN_FULL_LEADERBOARD_REPLAY = True')
start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l136-complete-replay-') as tmp:
 root=Path(tmp)/'l136-replay'
 shutil.copytree(P/'sources/l136',root/'sources/l136')
 shutil.copytree(P/'results/l136/leaderboard-data',root/'results/l136/leaderboard-data')
 NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 r=json.loads((root/'evidence/l136/leaderboard.json').read_text());expected=json.loads((P/'evidence/l136/leaderboard.json').read_text());assert r==expected
report=dict(status='PASS',code_sha256=digest,seconds=time.perf_counter()-start,code_cells=sum(c.cell_type=='code' for c in nb.cells),gate='RUN_FULL_LEADERBOARD_REPLAY=True; training remains False',predictions_rescored=r['predictions_rescored'],task_entry_pairs=27,cache='Copied pinned official files; no repository imports; no new training',result='EXACT_PRIMARY_AUDIT')
(P/'_notebook_replay_l136_results.json').write_text(json.dumps(report,indent=2));print(report)
