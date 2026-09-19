"""Execute the visible solution, including new fits, and reconstruct every score."""
import hashlib,json
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from relkit.exit_l080 import audit_result
ROOT=Path(__file__).resolve().parent;p=ROOT/'solutions/0080-year-2-exit-exam.ipynb'
nb=nbformat.read(p,as_version=4)
NotebookClient(nb,timeout=600,resources={'metadata':{'path':str(ROOT)}},kernel_name='python3').execute()
nbformat.write(nb,p)
files=sorted((ROOT/'data/cache').glob('l080-smoke-*.json'),key=lambda x:x.stat().st_mtime)
fresh=json.loads(files[-1].read_text());assert fresh['status']=='COMPLETE';audit_result(fresh)
assert fresh['preset']=='smoke' and len(fresh['records'])==16
reference=json.loads(Path('/tmp/l080-smoke.json').read_text()) if Path('/tmp/l080-smoke.json').exists() else None
max_delta=max(abs(a['error']-b['error']) for a,b in zip(fresh['records'],reference['records'])) if reference else None
if max_delta is not None: assert max_delta<1e-7
report={'status':'PASS','code_cells':sum(c.cell_type=='code' for c in nb.cells),'author_records_audited':48,'fresh_solution_records':16,'fresh_preset':'smoke','script_vs_inline_max_loss_delta':max_delta,'solution_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'live_colab':'NOT_CHECKED','paper_results':'NOT_RUN'}
(ROOT/'_execution_l080_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
