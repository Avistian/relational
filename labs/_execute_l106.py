"""Run solution in clean directory, rehash raw input, recalculate every prediction."""
import hashlib,json,os,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='0106-temporal-link-prediction';path=P/'solutions'/f'{S}.ipynb'
nb=nbformat.read(path,as_version=4);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l106-notebook-') as tmp:
 w=Path(tmp);(w/'l106-data').mkdir();os.link(P/'data/l102/wikipedia.csv',w/'l106-data/wikipedia.csv')
 NotebookClient(nb,timeout=1200,kernel_name='python3',resources={'metadata':{'path':str(w)}}).execute()
 fresh=json.loads((w/'l106-fresh.json').read_text())
nbformat.write(nb,path);html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
r={'status':'PASS','seconds':time.perf_counter()-start,'scope':'All visible code; all test events, six conditions, five loops; raw data rehashed; frozen source candidates authenticated','candidate_regeneration':'NOT_RUN in notebook; separately executed by author runner','fresh':fresh,'code_cells':sum(c.cell_type=='code' for c in nb.cells),'executed_code_sha256':hashlib.sha256('\n\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest(),'live_colab':'NOT_CHECKED','learner_status':'PENDING_WRITTEN_DEFENSE'}
(P/'_execution_l106_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
