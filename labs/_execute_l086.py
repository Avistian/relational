"""Execute portable solution in an empty working directory and compare seed0."""
from pathlib import Path
import json,tempfile,hashlib
import nbformat
from nbclient import NotebookClient
LAB=Path(__file__).resolve().parent;p=LAB/'solutions/0086-pyg-fundamentals.ipynb'
nb=nbformat.read(p,as_version=4)
with tempfile.TemporaryDirectory(prefix='l086-notebook-') as tmp:
 NotebookClient(nb,timeout=1200,resources={'metadata':{'path':tmp}},kernel_name='python3').execute()
 result=json.loads((Path(tmp)/'l086-exit.json').read_text())
 expected=json.loads((LAB/'_paper_l086_results.json').read_text())['runs'][0]
 assert result['runs']==[expected], 'Notebook trainer differs from canonical run'
nbformat.write(nb,p)
r={'status':'PASS','code_cells':sum(c.cell_type=='code' for c in nb.cells),'fresh_download_and_empty_cwd':True,'seeds_executed':1,'exact_author_seed0_trace':True,'solution_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'live_colab':'NOT_CHECKED'}
(LAB/'_execution_l086_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
