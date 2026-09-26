"""Execute standalone solution in a clean directory with authenticated raw-input reuse."""
import hashlib,json,os,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='0107-snapshot-methods';path=P/'solutions'/f'{S}.ipynb'
nb=nbformat.read(path,as_version=4);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l107-notebook-') as tmp:
 w=Path(tmp)
 for dest,source in [('l107-data/wiki/wikipedia.csv','data/l102/wikipedia.csv'),('l107-data/sbm/sbm.csv','data/l107/sbm.csv')]:
  target=w/dest;target.parent.mkdir(parents=True,exist_ok=True);os.link(P/source,target)
 NotebookClient(nb,timeout=1200,kernel_name='python3',resources={'metadata':{'path':str(w)}}).execute()
 fresh=json.loads((w/'l107-fresh.json').read_text())
nbformat.write(nb,path);html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
r={'status':'PASS','seconds':time.perf_counter()-start,'scope':'Standalone full-data daily ten-epoch fit plus full-size SBM pilot; full named paper lane gated off','fresh':fresh,'code_cells':sum(c.cell_type=='code' for c in nb.cells),'executed_code_sha256':hashlib.sha256('\n\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest(),'live_colab':'NOT_CHECKED','learner_status':'PENDING_WRITTEN_DEFENSE'}
(P/'_execution_l107_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
