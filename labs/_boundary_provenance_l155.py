"""Authenticate old full-training runs while updating their reporting helper."""
import gzip,hashlib,json
from pathlib import Path
import nbformat
P=Path(__file__).resolve().parent

def check():
    from _boundary_provenance_l152 import check as check_dependency
    check_dependency()
    old=gzip.decompress((P/'sources/l155/notebook_before_boundary.py.gz').read_bytes()).decode()
    b=nbformat.read(P/'solutions/0155-compare-manual-fe.ipynb',4)
    new='\n\n'.join(c.source for c in b.cells if c.cell_type=='code')
    assert old.count('<=.20 else')==2
    assert new==old.replace('<=.20 else','<=.20+8*math.ulp(4.022) else')
    digest=hashlib.sha256(old.encode()).hexdigest()
    for name in ('_notebook_gnn_l155_results.json','_notebook_fe_l155_results.json'):
        assert json.loads((P/name).read_text())['code_sha256']==digest
    return dict(status='PASS',historical_code_sha256=digest,current_code_sha256=hashlib.sha256(new.encode()).hexdigest(),scope='Only reporting helper and its embedded copy changed; all training and data code identical',full_training='ARCHIVED_PASS; not rerun for reporting correction')

if __name__=='__main__':print(check())
