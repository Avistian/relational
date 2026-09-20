"""Execute every inline solution cell with a fresh local data directory."""
import json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
P=Path(__file__).resolve().parent;path=P/'solutions/0091-r-gcn.ipynb';nb=nbformat.read(path,as_version=4)
start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l091-inline-') as tmp:
 NotebookClient(nb,timeout=1200,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 paper=json.loads((Path(tmp)/'l091-paper.json').read_text());basis=json.loads((Path(tmp)/'l091-basis.json').read_text())
 author=json.loads((P/'_paper_l091_results.json').read_text())
 assert paper['data']['node_order_sha256']==author['data']['node_order_sha256']
 assert [r['test_predictions'] for r in paper['runs']]==[r['test_predictions'] for r in author['runs']]
 assert [r['train_loss'] for r in paper['runs']]==[r['train_loss'] for r in author['runs']]
 (P/'_inline_paper_l091_results.json').write_text(json.dumps(paper,indent=2)+'\n')
 (P/'_inline_basis_l091_results.json').write_text(json.dumps(basis,indent=2)+'\n')
nbformat.write(nb,path)
result={'status':'PASS','executed_code_cells':sum(c.cell_type=='code' for c in nb.cells),'inline_paper_runs':len(paper['runs']),'inline_basis_runs':len(basis['runs']),'fresh_directory_download':'PASS','author_prediction_and_loss_trace_equality':'PASS','seconds':time.perf_counter()-start,'live_colab':'NOT_CHECKED'}
(P/'_execution_l091_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
