"""Execute the inline notebook and compare every score/loss trace to canonical fresh training."""
import hashlib,json
from pathlib import Path
import nbformat
from nbclient import NotebookClient
LAB=Path(__file__).resolve().parent;p=LAB/'solutions/0083-graphsage.ipynb'
nb=nbformat.read(p,as_version=4)
NotebookClient(nb,timeout=1800,resources={'metadata':{'path':str(LAB)}},kernel_name='python3').execute();nbformat.write(nb,p)
a=json.loads((LAB/'_paper_l083_results.json').read_text());b=json.loads((LAB/'l083-student-results.json').read_text())
for left,right in zip(a['runs'],b['runs']):
    for r in [left,right]:
        for c in r['candidates']:c.pop('seconds')
assert a['runs']==b['runs'] and len(b['runs'])==3
report={'status':'PASS','code_cells':sum(c.cell_type=='code' for c in nb.cells),'full_seed_searches':3,'all_scores_and_epoch_losses_match':True,'notebook_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'live_colab':'NOT_CHECKED'}
(LAB/'_execution_l083_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
