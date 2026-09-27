"""Execute the portable solution in an empty directory, without repository imports."""
import hashlib,json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='0130-rdl-checkpoint';path=P/'solutions'/f'{S}.ipynb';nb=nbformat.read(path,as_version=4);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l130-notebook-') as tmp:
 NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 report=json.loads((Path(tmp)/'l130-report.json').read_text());assert report['status']=='PASS' and report['predictions']==6295
 trace=json.loads((Path(tmp)/'l130-trace.json').read_text());assert trace['nodes']==99 and trace['edges']==245
 (P/'evidence/l130/trace.json').write_text(json.dumps(trace,indent=2))
 (P/'_teaching_l130_results.json').write_text(json.dumps(report,indent=2))
nbformat.write(nb,path);html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
r=dict(status='PASS',seconds=time.perf_counter()-start,code_cells=sum(c.cell_type=='code' for c in nb.cells),executed_code_sha256=hashlib.sha256('\n\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest(),default_run='Three live experiment functions, 6295 predictions and full-graph real-query reconstruction; no repo dependencies',full_training_gate='OFF; separate author GPU evidence',live_colab='NOT_CHECKED',learner_status='PENDING_WRITTEN_DEFENSE')
(P/'_execution_l130_results.json').write_text(json.dumps(r,indent=2));print(r)
