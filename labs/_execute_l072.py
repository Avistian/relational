"""Execute all visible solution cells, including fresh fits, and compare records."""
import json,hashlib
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parent;SLUG='0072-scarf-subtab-contrastive-views'
if __name__=='__main__':
    p=ROOT/'solutions'/f'{SLUG}.ipynb';nb=nbformat.read(p,as_version=4)
    NotebookClient(nb,timeout=3600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,p)
    actual=json.loads((ROOT/'l072-student-results.json').read_text());expected=json.loads((ROOT/'_verify_l072_results.json').read_text())
    assert actual['records']==expected['records'], 'Notebook live implementation differs from reference'
    report={'status':'PASS','code_cells':sum(c.cell_type=='code' for c in nb.cells),'evaluations':len(actual['records']),
            'exact_record_parity':True,'implementation_sha256':hashlib.sha256((ROOT/'relkit/contrastive_l072.py').read_bytes()).hexdigest(),
            'live_colab':'NOT_CHECKED','larger_run':'NOT_RUN'}
    (ROOT/'_execution_l072_results.json').write_text(json.dumps(report,indent=2));(ROOT/'l072-student-results.json').unlink();print(report)
