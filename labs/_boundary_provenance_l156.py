"""Separate archived full training from current reporting-only corrections."""
import gzip,hashlib,json
from pathlib import Path
import nbformat
P=Path(__file__).resolve().parent

def check():
    from _boundary_provenance_l152 import check as check_dependency
    check_dependency()
    old=gzip.decompress((P/'sources/l156/notebook_before_boundary.py.gz').read_bytes()).decode()
    b=nbformat.read(P/'solutions/0156-temporal-leakage-audit.ipynb',4)
    new='\n\n'.join(c.source for c in b.cells if c.cell_type=='code')
    assert old.count('<=.20 else')==1 and old.count('<=.2 else')==1
    assert new==old.replace('<=.20 else','<=.20+8*math.ulp(4.022) else').replace('<=.2 else','<=.2+8*math.ulp(target) else')
    replay=(P/'sources/l156/replay_before_boundary.py').read_text()
    assert (P/'_replay_l156.py').read_text()==replay.replace('import hashlib,json,statistics','import hashlib,json,statistics,math').replace('<=.2 else','<=.2+8*math.ulp(target) else')
    historical=json.loads((P/'evidence/l156/notebook/execution.json').read_text())
    digest=hashlib.sha256(old.encode()).hexdigest();assert digest==historical['code_sha256']
    return dict(status='PASS',historical_code_sha256=digest,current_code_sha256=hashlib.sha256(new.encode()).hexdigest(),scope='Two inclusive reporting comparisons only; training and data code unchanged',full_training='ARCHIVED_PASS; not rerun after reporting fix')

if __name__=='__main__':print(check())
