"""Fresh standalone solution execution; no course imports, caches or score injection."""
import hashlib,json,os,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='0109-database-timestamp-contracts';path=P/'solutions'/f'{S}.ipynb'
nb=nbformat.read(path,as_version=4);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l109-notebook-') as tmp:
    NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
    fresh=json.loads((Path(tmp)/'l109-fresh.json').read_text())
nbformat.write(nb,path);html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
r={'status':'PASS','seconds':time.perf_counter()-start,'fresh':fresh,'code_cells':sum(c.cell_type=='code' for c in nb.cells),'executed_code_sha256':hashlib.sha256('\n\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest(),'scope':'Three live contract tasks, all labels and ten paper cells; fresh archive download; one real graph snapshot','source_sql_full_graph_census':'SEPARATE_AUTHOR_CHECKS','live_colab':'NOT_CHECKED','learner_status':'PENDING_WRITTEN_DEFENSE'}
(P/'_execution_l109_results.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='fresh'})
