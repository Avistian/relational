"""Execute all inline cells from a temporary directory and compare every full run."""
import hashlib,json,shutil,tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
P=Path(__file__).resolve().parent;path=P/'solutions/0097-negative-sampling.ipynb';nb=nbformat.read(path,as_version=4)
with tempfile.TemporaryDirectory(prefix='l097-inline-') as folder:
 data=Path(folder)/'l097-data';data.mkdir();shutil.copy2(P/'data/l095/ml-100k.zip',data/'ml-100k.zip')
 NotebookClient(nb,timeout=300,kernel_name='relational-labs',resources={'metadata':{'path':folder}}).execute()
 actual=json.loads(Path(folder,'l097-fresh.json').read_text());expected=json.loads((P/'_experiment_l097_results.json').read_text())
 assert actual['runs']==expected['runs'] and actual['summary']==expected['summary']
nbformat.write(nb,path)
result={'status':'PASS','code_cells':sum(c.cell_type=='code' for c in nb.cells),'fresh_fits':45,'all_run_records':'EXACT','source_sha256':hashlib.sha256((P/'relkit/negative_l097.py').read_bytes()).hexdigest(),'live_colab':'NOT_CHECKED'}
(P/'_execution_l097_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
