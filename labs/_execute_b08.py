"""Execute portable B08 solution from an empty directory; compare all fresh metrics."""
import json,tempfile,time,hashlib
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
P=Path(__file__).resolve().parent;S='b08-structured-objectives';path=P/'solutions'/(S+'.ipynb');book=nbformat.read(path,4);start=time.monotonic()
with tempfile.TemporaryDirectory(prefix='b08-solution-') as td:
 NotebookClient(book,timeout=600,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
 report=json.loads((Path(td)/'b08-report.json').read_text());assert report['fresh'];assert report['all_prediction_arrays']=='EXACT'
 audit=json.loads((P/'evidence/b08/course-audit.json').read_text())
 for row in report['metrics']:
  old=next(r for r in audit['rows'] if (r['seed'],r['objective'])==(row['seed'],row['objective']))
  for key in ['target_mse','feature_mse']:assert abs(row[key]-old[key])<1e-12
 assert json.loads((Path(td)/'b08-submission.json').read_text())['status']=='PENDING_WRITTEN_DEFENSE'
 (P/'evidence/b08/solution-report.json').write_text(json.dumps(report,indent=2)+'\n')
nbformat.write(book,path)
exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True)
html,_=exporter.from_notebook_node(book);(P/'html'/(S+'.html')).write_text(html)
r=dict(status='PASS',code_cells=sum(c.cell_type=='code' for c in book.cells),working_directory='EMPTY_TEMPORARY',fresh_prediction_arrays='EXACT',seconds=time.monotonic()-start,executed_code_sha256=hashlib.sha256('\n\n'.join(c.source for c in book.cells if c.cell_type=='code').encode()).hexdigest(),live_colab='NOT_CHECKED')
(P/'_execution_b08_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
