"""Complete standalone replay in fresh directory; compare all non-timing numerical evidence."""
import hashlib,json,tempfile,subprocess
from pathlib import Path
import nbformat
from nbclient import NotebookClient
P=Path(__file__).resolve().parent;path=P/'solutions/0099-rgcn-vs-hgt.ipynb';nb=nbformat.read(path,as_version=4)
with tempfile.TemporaryDirectory(prefix='l099-inline-') as folder:
 data=Path(folder,'l099-data');data.mkdir()
 subprocess.run(['curl','-L','--fail','--retry','2','--connect-timeout','10','--max-time','60','-sS','-o',str(data/'ACM.mat'),'https://data.dgl.ai/dataset/ACM.mat'],check=True)
 assert hashlib.sha256((data/'ACM.mat').read_bytes()).hexdigest()=='0ccd838e545e8f16e3dc84356da2f51dfd2290c32e37a784e374dd510c76578d'
 NotebookClient(nb,timeout=600,kernel_name='relational-labs',resources={'metadata':{'path':folder}}).execute()
 fresh=json.loads(Path(folder,'l099-fresh.json').read_text());author=json.loads((P/'_experiment_l099_results.json').read_text())
 assert fresh['data']==author['data'] and fresh['selected_rates']==author['selected_rates']
 for a,b in zip(fresh['runs'],author['runs']):
  assert {k:v for k,v in a.items() if k not in ('seconds','checkpoint_sha256')}=={k:v for k,v in b.items() if k not in ('seconds','checkpoint_sha256')}
 for a,b in zip(fresh['selected'],author['selected']):
  assert {k:v for k,v in a.items() if k!='seconds'}=={k:v for k,v in b.items() if k!='seconds'}
 assert fresh['paired_accuracy_differences']==author['paired_accuracy_differences']
 (P/'_inline_l099_results.json').write_text(json.dumps(fresh,indent=2)+'\n')
nbformat.write(nb,path)
report={'status':'PASS','code_cells':sum(c.cell_type=='code' for c in nb.cells),'fresh_fits':24,'fresh_epochs':1440,'all_loss_traces_and_selected_predictions':'EXACT','working_directory':'fresh temporary directory, harness independently downloads via bounded curl and notebook verifies hash, no repository imports','environment':'same author kernel; clean environment build NOT_CHECKED','inline_source_sha256':hashlib.sha256(''.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest(),'live_colab':'NOT_CHECKED'}
(P/'_execution_l099_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
