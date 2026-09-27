"""Execute portable solution in an empty directory, including full feature export."""
import hashlib,json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
import numpy as np
P=Path(__file__).resolve().parent;S='0125-pytorch-frame-deep-dive';path=P/'solutions'/f'{S}.ipynb';nb=nbformat.read(path,as_version=4);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l125-notebook-') as tmp:
 NotebookClient(nb,timeout=240,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 report=json.loads((Path(tmp)/'l125-feature-report.json').read_text());assert report['status']=='PASS' and report['total_rows']==74063
 actual=np.load(Path(tmp)/'l125-encoded-reg.npz',allow_pickle=False);expected=np.load(P/'evidence/l125/encoded-reg.npz',allow_pickle=False)
 for name in expected.files:np.testing.assert_array_equal(actual[name],expected[name])
 (P/'_teaching_l125_results.json').write_text(json.dumps(report,indent=2)+'\n')
nbformat.write(nb,path);html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
r={'status':'PASS','seconds':time.perf_counter()-start,'code_cells':sum(c.cell_type=='code' for c in nb.cells),'executed_code_sha256':hashlib.sha256('\n\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest(),'default_run':'Four live TODOs, pinned Frame model parity, all 74063 feature rows, 23 real training-query gradient step','all_export_arrays':'EXACT_MATCH','repository_imports':False,'historical_paper_training':'NOT_RUN','live_colab':'NOT_CHECKED','learner_status':'PENDING_WRITTEN_DEFENSE'}
(P/'_execution_l125_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
