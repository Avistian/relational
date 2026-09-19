"""Execute the live solution, compare canonical diagnostic, preserve outputs."""
import hashlib,json
from pathlib import Path
import nbformat
from nbclient import NotebookClient
LAB=Path(__file__).resolve().parent;SLUG='0077-single-table-ceiling'
def run():
 p=LAB/'solutions'/f'{SLUG}.ipynb';nb=nbformat.read(p,as_version=4)
 NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(LAB)}}).execute()
 nbformat.write(nb,p)
 actual=json.loads((LAB/'l077-student-results.json').read_text());expected=json.loads((LAB/'_verify_l077_results.json').read_text())['runs']
 assert actual==expected,'Inline notebook and canonical experiment differ'
 report={'status':'PASS','code_cells':sum(c.cell_type=='code' for c in nb.cells),'exact_diagnostic_parity':True,
 'implementation_sha256':hashlib.sha256((LAB/'relkit/ceiling_l077.py').read_bytes()).hexdigest(),'full_synthetic_reproduction':'MATCH','live_colab':'NOT_CHECKED'}
 (LAB/'_execution_l077_results.json').write_text(json.dumps(report,indent=2)+'\n');(LAB/'l077-student-results.json').unlink();print(report)
if __name__=='__main__':run()
