"""Execute the standalone inline solution in a fresh directory; compare measured runs."""
import argparse,json,os,tempfile,hashlib
from pathlib import Path
import nbformat
from nbclient import NotebookClient
P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--teaching',action='store_true');args=parser.parse_args()
path=P/'solutions/0093-hgt.ipynb';nb=nbformat.read(path,as_version=4)
os.environ['L093_DATA_DIR']=str(P/'data/l093');os.environ['L093_RUN_TEACHING']='1' if args.teaching else '0'
with tempfile.TemporaryDirectory(prefix='l093-inline-') as directory:
 client=NotebookClient(nb,timeout=1800,kernel_name='python3',resources={'metadata':{'path':directory}})
 client.execute();nbformat.write(nb,path)
 result={'status':'PASS','scope':'inline diagnostic, tasks and interventions'+('; nine fresh comparison fits' if args.teaching else ''),'code_cells':sum(c.cell_type=='code' for c in nb.cells),'all_code_cells_executed':all(c.execution_count is not None for c in nb.cells if c.cell_type=='code'),'source_sha256':hashlib.sha256((P/'relkit/hgt_l093.py').read_bytes()).hexdigest(),'live_colab':'NOT_CHECKED','CS_reproduction':'NOT_RUN'}
 if args.teaching:
  actual=json.loads(Path(directory,'l093-inline-teaching.json').read_text());expected=json.loads((P/'_teaching_l093_results.json').read_text())
  errors=[]
  for a,b in zip(actual['runs'],expected['runs']):
   assert (a['seed'],a['arm'])==(b['seed'],b['arm']);assert a['selected_epoch']==b['selected_epoch']
   assert [q['paper_id'] for q in a['test_queries']]==[q['paper_id'] for q in b['test_queries']]
   for key in ('test_ndcg','test_mrr'):
    delta=abs(a[key]-b[key]);errors.append(delta);assert delta<2e-6,(a['arm'],key,delta)
  (P/'_inline_l093_results.json').write_text(json.dumps(actual,indent=2)+'\n');result['replayed_fits']=len(actual['runs']);result['max_metric_difference']=max(errors);result['metric_tolerance']=2e-6
(P/'_execution_l093_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
