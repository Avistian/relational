"""Execute every solution cell in a fresh notebook kernel."""
import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parent
path=ROOT/'solutions/0081-mpnn-framework.ipynb'
nb=nbformat.read(path,as_version=4)
NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
nbformat.write(nb,path)
exit_file=ROOT/'l081-exit.json'
if exit_file.exists():
    (ROOT/'evidence/l081/author-exit.json').write_bytes(exit_file.read_bytes());exit_file.unlink()
report={'status':'PASS','executed_code_cells':sum(c.cell_type=='code' for c in nb.cells),'live_colab':'NOT_CHECKED'}
(ROOT/'_execution_l081_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
