"""Execute the entire standalone notebook away from the repository."""
import hashlib,json,tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
P=Path(__file__).resolve().parent;path=P/'solutions/0096-multi-relational-data.ipynb';nb=nbformat.read(path,as_version=4)
with tempfile.TemporaryDirectory(prefix='l096-inline-') as folder:
    NotebookClient(nb,timeout=120,kernel_name='relational-labs',resources={'metadata':{'path':folder}}).execute()
    actual=json.loads(Path(folder,'l096-fresh.json').read_text())
    expected=json.loads((P/'_experiment_l096_results.json').read_text())
    assert actual['records']==expected['records']
nbformat.write(nb,path)
result={'status':'PASS','code_cells':sum(c.cell_type=='code' for c in nb.cells),'fresh_databases':33,'all_records':'EXACT','source_sha256':hashlib.sha256((P/'relkit/schema_l096.py').read_bytes()).hexdigest(),'live_colab':'NOT_CHECKED'}
(P/'_execution_l096_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
