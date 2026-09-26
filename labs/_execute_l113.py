"""Execute the solution in a clean directory; preserve student blanks."""
import hashlib,json,os,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='0113-scaling-ogb';path=P/'solutions'/f'{S}.ipynb';nb=nbformat.read(path,as_version=4);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l113-notebook-') as tmp:
 d=Path(tmp)/'l113-arxiv';d.mkdir();os.symlink(P/'data/l112/arxiv.zip',d/'arxiv.zip')
 NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 fresh=json.loads((Path(tmp)/'l113-fresh.json').read_text())
nbformat.write(nb,path)
html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}-solution.html').write_text(html)
code='\n\n'.join(c.source for c in nb.cells if c.cell_type=='code')
r={'status':'PASS','seconds':time.perf_counter()-start,'executed_cells':sum(c.cell_type=='code' for c in nb.cells),'executed_code_sha256':hashlib.sha256(code.encode()).hexdigest(),'fresh_bridge':fresh,'full_products_in_kernel':'NOT_RUN','live_colab':'NOT_CHECKED'}
(P/'_execution_l113_results.json').write_text(json.dumps(r,indent=2));print(r)
