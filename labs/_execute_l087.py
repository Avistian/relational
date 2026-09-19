"""Execute the self-contained solution from an empty directory with fresh data."""
from pathlib import Path
import tempfile,json,hashlib
import nbformat
from nbclient import NotebookClient
LAB=Path(__file__).resolve().parent;p=LAB/'solutions/0087-link-prediction.ipynb';nb=nbformat.read(p,as_version=4)
with tempfile.TemporaryDirectory(prefix='l087-notebook-') as tmp:
    NotebookClient(nb,timeout=600,resources={'metadata':{'path':tmp}},kernel_name='python3').execute()
    actual=json.loads((Path(tmp)/'l087-exit.json').read_text());expected=json.loads((LAB/'_teaching_l087_results.json').read_text())['runs'][0]
    expected.pop('artifact_sha256');assert actual['teaching']==expected,'Notebook implementation diverged'
    full=json.loads((Path(tmp)/'l087-paper-results.json').read_text());paper=json.loads((LAB/'_paper_l087_results.json').read_text())
    assert full['summary']==paper['summary'] and full['statistics']==paper['statistics']
    assert len(full['runs'])==80
nbformat.write(nb,p)
r={'status':'PASS','code_cells':sum(c.cell_type=='code' for c in nb.cells),'empty_cwd_fresh_data':True,'full_teaching_trace_exact_match':True,'full_baseline_track_in_this_kernel':240,'baseline_summary_exact_match':True,'solution_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'live_colab':'NOT_CHECKED'}
(LAB/'_execution_l087_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
