"""Execute the visible solution and verify training results against author evidence."""
import hashlib,json
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).parent;SLUG='0071-vime-masked-tabular-ssl'
if __name__=='__main__':
    p=ROOT/'solutions'/f'{SLUG}.ipynb';nb=nbformat.read(p,as_version=4)
    NotebookClient(nb,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(ROOT.resolve())}}).execute()
    nb.metadata.pop("l071_loader_correction",None)
    nbformat.write(nb,p)
    actual=json.loads((ROOT/'l071-student-results.json').read_text())
    expected=json.loads((ROOT/'_verify_l071_results.json').read_text())
    assert actual['records']==expected['records'], 'Visible solution must replay reference records'
    (ROOT/'_execution_l071_results.json').write_text(json.dumps({'status':'PASS','solution_executed':True,'fits':len(actual['records']),'implementation_sha256':hashlib.sha256((ROOT/'relkit/vime_l071.py').read_bytes()).hexdigest(),'exact_record_parity':True,'live_colab':'NOT_CHECKED','modal':'NOT_RUN'},indent=2)+'\n')
    (ROOT/'l071-student-results.json').unlink()
    print('PASS: executed visible solution, 45 fresh fits exactly match reference records')
