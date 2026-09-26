"""Execute visible notebook in a clean directory; no hidden imports from checkout."""
import hashlib,json,os,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='0108-temporal-neighbor-sampling';path=P/'solutions'/f'{S}.ipynb'
nb=nbformat.read(path,as_version=4);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l108-notebook-') as tmp:
 w=Path(tmp);(w/'l108-data').mkdir()
 for name in ['wikipedia.csv','processed.npz']:os.link(P/'l103-cache'/name,w/'l108-data'/name)
 NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(w)}}).execute()
 fresh=json.loads((w/'l108-fresh.json').read_text())
nbformat.write(nb,path);html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
r={'status':'PASS','seconds':time.perf_counter()-start,'scope':'Three live tasks; synthetic model handoff; all 157474 full-data sampler queries with equality and repeated benchmark; raw and processed bytes rehashed','full_gpu_evaluation':'SEPARATE_AUTHOR_RUN','fresh':fresh,'code_cells':sum(c.cell_type=='code' for c in nb.cells),'executed_code_sha256':hashlib.sha256('\n\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest(),'live_colab':'NOT_CHECKED','learner_status':'PENDING_WRITTEN_DEFENSE'}
(P/'_execution_l108_results.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='fresh'})
