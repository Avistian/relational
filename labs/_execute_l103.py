"""Execute the standalone visible solution and render a prepared notebook view."""
import hashlib,json,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='0103-tgat';cache=P/'l103-cache';cache.mkdir(exist_ok=True)
for name in ['wikipedia.csv','processed.npz']:
 target=cache/name
 if not target.exists():target.symlink_to(P/'data/l102'/name)
path=P/'solutions'/f'{S}.ipynb';nb=nbformat.read(path,as_version=4);begin=time.perf_counter()
NotebookClient(nb,timeout=1200,kernel_name='python3',resources={'metadata':{'path':str(P)}}).execute()
nbformat.write(nb,path);html,_=HTMLExporter(template_name='lab').from_notebook_node(nb)
(P/'html'/f'{S}.html').write_text(html)
report={'status':'PASS','code_cells':sum(c.cell_type=='code' for c in nb.cells),'seconds':time.perf_counter()-begin,'notebook_sha256_at_execution':hashlib.sha256(path.read_bytes()).hexdigest(),'executed_code_sha256':hashlib.sha256('\n\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest(),'scope':'Fresh 600/120/120 event, one-epoch matched comparison and all live tasks; full replay gated off','live_colab':'NOT_CHECKED'}
(P/'_execution_l103_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
