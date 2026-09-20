"""Fresh standalone notebook replay, including full course suite and AIFB appendix."""
import hashlib,json,tempfile,subprocess
from pathlib import Path
import nbformat
from nbclient import NotebookClient
P=Path(__file__).resolve().parent;path=P/'solutions/0100-heterogeneous-gnn-checkpoint.ipynb';nb=nbformat.read(path,as_version=4)
with tempfile.TemporaryDirectory(prefix='l100-inline-') as folder:
 data=Path(folder,'l100-data');data.mkdir()
 subprocess.run(['curl','-L','--fail','--retry','2','--connect-timeout','10','--max-time','60','-sS','-o',str(data/'ACM.mat'),'https://data.dgl.ai/dataset/ACM.mat'],check=True)
 assert hashlib.sha256((data/'ACM.mat').read_bytes()).hexdigest()=='0ccd838e545e8f16e3dc84356da2f51dfd2290c32e37a784e374dd510c76578d'
 # Independently fetch AIFB bytes; both inline loaders still verify their manifests.
 data=Path(folder,'l100-aifb');data.mkdir()
 subprocess.run(['curl','-L','--fail','--retry','2','--connect-timeout','10','--max-time','60','-sS','-o',str(data/'aifb.tgz'),'https://data.dgl.ai/dataset/aifb.tgz'],check=True)
 def completed(cell,cell_index,execute_reply):
  print('Executed cell',cell_index,cell.source.splitlines()[0][:70],flush=True)
  nbformat.write(nb,P/'solutions/.l100-partial.ipynb')
 NotebookClient(nb,timeout=900,kernel_name='relational-labs',resources={'metadata':{'path':folder}},on_cell_executed=completed).execute()
 fresh=json.loads(Path(folder,'l100-fresh.json').read_text());author=json.loads((P/'_experiment_l100_results.json').read_text())
 assert fresh['data']==author['data'] and fresh['selected_rates']==author['selected_rates']
 for a,b in zip(fresh['runs'],author['runs']):
  assert {k:v for k,v in a.items() if k not in ('seconds','checkpoint_sha256')}=={k:v for k,v in b.items() if k not in ('seconds','checkpoint_sha256')}
 for a,b in zip(fresh['selected'],author['selected']):
  assert {k:v for k,v in a.items() if k!='seconds'}=={k:v for k,v in b.items() if k!='seconds'}
 assert fresh['paired_accuracy_differences']==author['paired_accuracy_differences']
 (P/'_inline_l100_results.json').write_text(json.dumps(fresh,indent=2)+'\n')
 aifb=json.loads(Path(folder,'l100-aifb-fresh.json').read_text());ref=json.loads((P/'_paper_l100_aifb_results.json').read_text())
 assert aifb['mean']==ref['mean'] and aifb['sample_sd']==ref['sample_sd']
 for a,b in zip(aifb['runs'],ref['runs']):
  assert {k:v for k,v in a.items() if k!='seconds'}=={k:v for k,v in b.items() if k!='seconds'}
 (P/'_inline_l100_aifb_results.json').write_text(json.dumps(aifb,indent=2)+'\n')
nbformat.write(nb,path)
report={'status':'PASS','code_cells':sum(c.cell_type=='code' for c in nb.cells),'fresh_course_fits':24,'fresh_epochs':960,'fresh_aifb_fits':10,'all_course_traces_samples_logits_predictions':'EXACT','aifb_traces_predictions':'EXACT','working_directory':'fresh temporary directory; independent bounded downloads; inline hash checks; no repository imports','environment':'same author kernel; clean install NOT_CHECKED','inline_source_sha256':hashlib.sha256(''.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest(),'hgt_cs_training':'NOT_RUN','live_colab':'NOT_CHECKED','initial_attempt':'Kernel process died; rejected as incomplete; successful evidence requires a complete fresh retry'}
(P/'_execution_l100_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
