"""Execute the standalone solution from an empty directory and verify full report parity."""
import hashlib,json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='0185-causal-relational-data';path=P/'solutions'/(S+'.ipynb')
book=nbformat.read(path,4);start=time.monotonic()
with tempfile.TemporaryDirectory(prefix='l185-notebook-') as tmp:
 NotebookClient(book,timeout=120,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 assert json.loads((Path(tmp)/'l185-report.json').read_text())==json.loads((P/'evidence/l185/report.json').read_text())
 assert json.loads((Path(tmp)/'l185-submission.json').read_text())['learner']=='PENDING_WRITTEN_DEFENSE'
nbformat.write(book,path)
html,_=HTMLExporter(template_name='lab').from_notebook_node(book)
html=html.replace('https://avistian.github.io/relational/','../../')
(P/'html'/(S+'.html')).write_text(html)
result={'status':'PASS','working_directory':'EMPTY_TEMPORARY','report_parity':'EXACT','seconds':time.monotonic()-start,'code_cells':sum(c.cell_type=='code' for c in book.cells),'code_sha256':hashlib.sha256('\n\n'.join(c.source for c in book.cells if c.cell_type=='code').encode()).hexdigest(),'live_colab':'NOT_CHECKED'}
(P/'_execution_l185_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
