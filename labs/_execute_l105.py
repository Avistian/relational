"""Execute every solution cell in a clean workdir using authenticated raw bytes."""
import hashlib,json,os,sys,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='0105-continuous-time'
path=P/'solutions'/f'{S}.ipynb';nb=nbformat.read(path,as_version=4);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l105-notebook-') as tmp:
 work=Path(tmp);(work/'l105-data').mkdir();os.link(P/'data/l102/wikipedia.csv',work/'l105-data/wikipedia.csv')
 NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(work)}}).execute()
 fresh=json.loads((work/'l105-fresh.json').read_text());assert fresh==json.loads((P/'_analysis_l105_results.json').read_text())['records']
nbformat.write(nb,path);html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
report={'status':'PASS','scope':'Every code cell, all three student seams, complete raw Wikipedia file and all three widths, SQLite and grouped-timestamp oracles','clean_workdir':True,'input_delivery':'Existing raw bytes hardlinked read-only by convention; rehashed inside notebook, no processed cache','new_network_download':'NOT_RUN: authenticated existing raw file used','seconds':time.perf_counter()-start,'code_cells':sum(c.cell_type=='code' for c in nb.cells),'executed_code_sha256':hashlib.sha256('\n\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest(),'author_results':'EXACT','live_colab':'NOT_CHECKED','learner_status':'PENDING_WRITTEN_DEFENSE'}
(P/'_execution_l105_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
