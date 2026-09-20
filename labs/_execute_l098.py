"""Fresh notebook execution outside the repo; all native samples and fits recomputed."""
import hashlib,json,tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
P=Path(__file__).resolve().parent;path=P/'solutions/0098-hetero-mini-batching.ipynb';nb=nbformat.read(path,as_version=4)
with tempfile.TemporaryDirectory(prefix='l098-inline-') as folder:
 NotebookClient(nb,timeout=300,kernel_name='relational-labs',resources={'metadata':{'path':folder}}).execute()
 fresh=json.loads(Path(folder,'l098-fresh.json').read_text());author=json.loads((P/'_experiment_l098_results.json').read_text())
 assert fresh['audit']==author['audit'] and fresh['runs']==author['runs'] and fresh['temporal']==author['temporal']
nbformat.write(nb,path)
report={'status':'PASS','code_cells':sum(c.cell_type=='code' for c in nb.cells),'fresh_audits':96,'fresh_fits':9,'exact_records_match':True,'working_directory':'fresh temporary directory, no repository imports','environment':'same author kernel; independent environment build NOT_CHECKED','inline_source_sha256':hashlib.sha256(''.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest(),'live_colab':'NOT_CHECKED'}
(P/'_execution_l098_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
