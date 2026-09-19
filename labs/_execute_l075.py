"""Fresh solution execution and exact real-row evidence comparison."""
import hashlib,json
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parent;SLUG='0075-pytorch-frame-row-encoder'
if __name__=='__main__':
 p=ROOT/'solutions'/f'{SLUG}.ipynb';nb=nbformat.read(p,as_version=4)
 NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
 nbformat.write(nb,p)
 actual=json.loads((ROOT/'l075-student-results.json').read_text());expected=json.loads((ROOT/'_verify_l075_results.json').read_text())['real_table']
 assert actual==expected,'Visible notebook differs from canonical reference'
 report={'status':'PASS','code_cells':sum(c.cell_type=='code' for c in nb.cells),'exact_row_vector_parity':True,'implementation_sha256':hashlib.sha256((ROOT/'relkit/frame_l075.py').read_bytes()).hexdigest(),'live_colab':'NOT_CHECKED'}
 (ROOT/'_execution_l075_results.json').write_text(json.dumps(report,indent=2)+'\n')
 for name in ['l075-student-results.json','l075-row-encoder.pt','l075-schema.json']:
  (ROOT/name).unlink()
 print(report)
