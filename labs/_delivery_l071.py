"""Check notebook wiring, local links and the relevant copied Pages staging paths."""
import ast,base64,hashlib,json,shutil
from pathlib import Path
from urllib.parse import urlsplit,unquote
import nbformat
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1];LABS=ROOT/'labs';SLUG='0071-vime-masked-tabular-ssl'
def check():
    # Copy directories exactly as the workflow does; no symlinks hide missing copies.
    stage=Path('/tmp/l071-pages');stage.mkdir(exist_ok=True)
    for name in ['assets','lessons','reference']:
        shutil.copytree(ROOT/name,stage/name,dirs_exist_ok=True)
    for name in ['html','figures','relkit','reproductions']:
        shutil.copytree(LABS/name,stage/'labs'/name,dirs_exist_ok=True)
    for name in ['index.html','notebooks.html']:shutil.copy2(ROOT/name,stage/name)
    (stage/'modal').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'modal/l071_paper_repro.py',stage/'modal/l071_paper_repro.py')
    for name in [SLUG+'.ipynb','_verify_l071_results.json','_sources_l071.json','_execution_l071_results.json','_browser_l071_results.json','_smoke_l071_results.json','_mnist_access_l071_results.json','_loader_fix_l071_results.json','l071-reproduction.md','_run_l071.py']:
        shutil.copy2(LABS/name,stage/'labs'/name)
    evidence=json.loads((LABS/'_verify_l071_results.json').read_text())
    core=(LABS/'relkit/vime_l071.py').read_text()
    assert hashlib.sha256(core.encode()).hexdigest()==evidence.get('current_implementation_sha256',evidence['implementation_sha256'])
    if 'current_implementation_sha256' in evidence:
        patch=json.loads((LABS/'_loader_fix_l071_results.json').read_text())
        node=next(n for n in ast.parse(core).body if isinstance(n,ast.FunctionDef) and n.name=='load_data')
        reconstructed=core.replace(ast.get_source_segment(core,node),patch['old_loader'])
        assert hashlib.sha256(reconstructed.encode()).hexdigest()==evidence['implementation_sha256']
    checked=0
    for rel in ['lessons/'+SLUG+'.html','reference/'+SLUG+'.html','labs/html/'+SLUG+'.html']:
        p=stage/rel;soup=BeautifulSoup(p.read_text(),'html.parser')
        for tag,attr in [('a','href'),('img','src'),('link','href'),('script','src')]:
            for el in soup.find_all(tag):
                url=el.get(attr,'');parts=urlsplit(url)
                if not url or parts.scheme or parts.netloc:continue
                target=(p.parent/unquote(parts.path)).resolve() if parts.path else p
                assert target.exists(),(rel,url)
                if parts.fragment and target.suffix=='.html':
                    target_soup=soup if target==p else BeautifulSoup(target.read_text(),'html.parser')
                    assert target_soup.find(id=parts.fragment) or target_soup.find(id=unquote(parts.fragment)),(rel,'missing anchor',url)
                checked+=1
    student=nbformat.read(LABS/(SLUG+'.ipynb'),as_version=4)
    solutions=nbformat.read(LABS/'solutions'/(SLUG+'.ipynb'),as_version=4)
    cells=[c.source for c in student.cells if c.cell_type=='code']
    assert sum('raise NotImplementedError' in c for c in cells)==3
    assert not any('raise NotImplementedError' in c.source for c in solutions.cells if c.cell_type=='code')
    code='\n'.join(c for c in cells if '@colab-bootstrap' not in c and not c.startswith('%%writefile '))
    tree=ast.parse(code)
    for name in ['corrupt','pretext_loss','consistency_loss']:
        assert sum(isinstance(n,ast.FunctionDef) and n.name==name for n in tree.body)==1
        assert f"['{name}'] is {name}" in code
    images=0
    import re
    for c in student.cells:
        if c.cell_type=='markdown':
            assert 'attachment:' not in c.source
            for encoded in re.findall(r'data:image/png;base64,([^"\s]+)',c.source):
                assert base64.b64decode(encoded).startswith(b'\x89PNG');images+=1
    assert images==5
    assert not any(c.execution_count is not None for c in student.cells if c.cell_type=='code')
    assert all(c.execution_count is not None for c in solutions.cells if c.cell_type=='code'), 'Solution must be executed'
    out=dict(status='PASS',local_links=checked,portable_figures=images,student_live_tasks=3,copied_pages=True,
             browser='See _browser_l071_results.json',live_colab='NOT_CHECKED',deployment='NOT_RUN')
    (LABS/'_delivery_l071_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':check()
