"""Execute all default solution cells: teaching plus one complete fresh Cora seed.
The separate CLI executes 100 full seeds; this check independently replays seed 0.
"""
import hashlib,json
from pathlib import Path
import nbformat
from nbclient import NotebookClient
LAB=Path(__file__).resolve().parent;p=LAB/'solutions/0084-gat.ipynb'
nb=nbformat.read(p,as_version=4)
NotebookClient(nb,timeout=1800,resources={'metadata':{'path':str(LAB)}},kernel_name='python3').execute();nbformat.write(nb,p)
report=LAB/'_paper_l084_results.json'
a=json.loads(report.read_text())['runs'][0] if report.exists() else json.loads((LAB/'runs/l084/seed-000.json').read_text())
a.pop('seconds',None)
b=json.loads((LAB/'l084-inline-seed0.json').read_text());assert a==b
r={'status':'PASS','code_cells':sum(c.cell_type=='code' for c in nb.cells),'fresh_full_seeds':1,'teaching_epochs':5,'seed0_all_epoch_losses_and_score_match':True,'notebook_100_seed_lane':'NOT_RUN','cli_100_seed_lane':'See _paper_l084_results.json','notebook_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'live_colab':'NOT_CHECKED'}
(LAB/'_execution_l084_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
