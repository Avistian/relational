"""Execute portable inline solution from an empty working directory."""
from pathlib import Path
import json,tempfile,hashlib
import nbformat
from nbclient import NotebookClient
LAB=Path(__file__).resolve().parent;p=LAB/'solutions/0088-graph-classification.ipynb'
nb=nbformat.read(p,as_version=4)
with tempfile.TemporaryDirectory(prefix='l088-notebook-') as tmp:
 NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 exit_artifact=json.loads((Path(tmp)/'l088-exit.json').read_text());assert not exit_artifact['complete']
 assert [r['readout'] for r in exit_artifact['smoke']]==['sum','mean','max']
 assert not (Path(tmp)/'l088-paper-results.json').exists()
 data_hash=hashlib.sha256((Path(tmp)/'l088-data/MUTAG.txt').read_bytes()).hexdigest()
nbformat.write(nb,p)
r={'status':'PASS','empty_cwd':True,'fresh_hash_verified_data':data_hash,'code_cells':sum(c.cell_type=='code' for c in nb.cells),'live_smoke_readouts':3,'full_grid_in_notebook':'NOT_RUN','full_teaching_in_notebook':'NOT_RUN','exit_complete':False,'live_colab':'NOT_CHECKED','smoke':exit_artifact['smoke']}
(LAB/'_execution_l088_results.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='smoke'})
