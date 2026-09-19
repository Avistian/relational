"""Editorial delivery checks: every lesson, topology, and unchanged code/output pairs."""
import hashlib,json
from pathlib import Path
import nbformat
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]
BASELINE=ROOT/'labs/_solution_maps_baseline.json'
def digest(nb):
    cells=[dict(source=c.source,outputs=c.get('outputs',[]),execution_count=c.get('execution_count')) for c in nb.cells if c.cell_type=='code']
    return hashlib.sha256(json.dumps(cells,sort_keys=True).encode()).hexdigest()
def snapshot():
    result={}
    for n in range(49,71):
        p=next((ROOT/'labs').glob(f'{n:04}-*.ipynb'))
        for q in [p,ROOT/'labs/solutions'/p.name]:result[str(q.relative_to(ROOT))]=digest(nbformat.read(q,4))
    BASELINE.write_text(json.dumps(result,indent=2)+'\n')
def check():
    baseline=json.loads(BASELINE.read_text());report=[]
    for n in range(49,71):
        p=next((ROOT/'lessons').glob(f'{n:04}-*.html'));s=BeautifulSoup(p.read_text(),'html.parser')
        assert len(s.select('.solution-story'))==1,(n,'opening story missing')
        assert len(s.select('.solution-seam'))==4,(n,'four authored transitions required')
        assert len(s.select('.solution-map'))==(2 if n==49 else 1),(n,'full solution map missing')
        assert len(s.select('.solution-handoff'))==1
        ids=[x['id'] for x in s.select('[id]')]
        assert len(ids)==len(set(ids)),(n,'duplicate IDs')
        for a in s.select('.solution-route a'):
            assert a['href'][1:] in ids,(n,'broken reading route',a['href'])
        for folder in ['labs','labs/solutions']:
            path=ROOT/folder/(p.stem+'.ipynb');nb=nbformat.read(path,4)
            assert digest(nb)==baseline[str(path.relative_to(ROOT))],(n,'code/output pairing changed')
            assert sum(bool(c.metadata.get('solution_map')) for c in nb.cells)==(2 if n==49 else 1)
            assert sum(c.metadata.get('solution_story')=='opening' for c in nb.cells)==1
        report.append(dict(lesson=n,maps=len(s.select('.solution-map')),authored_transitions=4,code_and_saved_outputs='UNCHANGED'))
    print(json.dumps(dict(status='PASS',lessons=report),indent=2))
if __name__=='__main__':
    import sys
    snapshot() if '--snapshot' in sys.argv else check()
