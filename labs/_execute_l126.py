"""Run the notebook in an empty directory and compare every output prediction."""
import hashlib,json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
import pandas as pd
P=Path(__file__).resolve().parent;S='0126-relbench-beta';path=P/'solutions'/f'{S}.ipynb';nb=nbformat.read(path,as_version=4);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l126-notebook-') as tmp:
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
    report=json.loads((Path(tmp)/'l126-report.json').read_text());reference=json.loads((P/'evidence/l126/summary.json').read_text())
    assert all(report[k]==reference[k] for k in report)
    actual=pd.read_csv(Path(tmp)/'l126-predictions.csv');expected=pd.read_csv(P/'evidence/l126/f1-predictions.csv')
    pd.testing.assert_frame_equal(actual,expected,check_exact=True)
    (P/'_teaching_l126_results.json').write_text(json.dumps(report,indent=2)+'\n')
nbformat.write(nb,path);html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
r={'status':'PASS','seconds':time.perf_counter()-start,'code_cells':sum(c.cell_type=='code' for c in nb.cells),'executed_code_sha256':hashlib.sha256('\n\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest(),'default_run':'Four live TODOs, composed beta fixture, complete F1 API tour','all760_predictions':'EXACT_MATCH','repository_imports':False,'historical_full_contract':'NOT_RUN','live_colab':'NOT_CHECKED','learner_status':'PENDING_WRITTEN_DEFENSE'}
(P/'_execution_l126_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
