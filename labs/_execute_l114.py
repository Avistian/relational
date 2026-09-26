"""Execute standalone solution with a copied, checksum-verified full evidence bundle."""
import hashlib,json,shutil,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
P=Path(__file__).resolve().parent;S='0114-ogb-error-analysis';path=P/'solutions'/f'{S}.ipynb';nb=nbformat.read(path,as_version=4);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l114-notebook-') as tmp:
 shutil.copyfile(P/'evidence/l114/analysis-inputs.npz',Path(tmp)/'l114-analysis-inputs.npz')
 NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 report=json.loads((Path(tmp)/'l114-error-report.json').read_text())
 summary=json.loads((P/'evidence/l114/summary.json').read_text())
 for key in ['selected_validation_slice','same_rule_on_test']:
  for field in ['population','family','slice','n','gcn','mlp','delta_pp','discordance']:
   assert report[key][field]==summary[key][field],(key,field)
nbformat.write(nb,path);html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
r={'status':'PASS','seconds':time.perf_counter()-start,'code_cells':sum(c.cell_type=='code' for c in nb.cells),'executed_code_sha256':hashlib.sha256('\n\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest(),'full_census':'Every validation/test node, all10 seeds, both models','analysis_matches_author':'EXACT','full_training_gate':'OFF; separate author GPU evidence','live_colab':'NOT_CHECKED','learner_status':'PENDING_WRITTEN_DEFENSE'}
(P/'_execution_l114_results.json').write_text(json.dumps(r,indent=2));print(r)
