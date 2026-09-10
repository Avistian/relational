"""L056 artifact consistency, portable images, links and realistic copied staging."""
import base64,hashlib,json,re,shutil,tempfile
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit,unquote
import nbformat
from PIL import Image
from bs4 import BeautifulSoup
from _build_l056 import ROOT,SLUG,build

site=ROOT.parent
result=json.loads((ROOT/'_verify_l056_results.json').read_text())
for p,sha in result['source_hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha,p
student=nbformat.read(ROOT/(SLUG+'.ipynb'),as_version=4)
solution=nbformat.read(ROOT/'solutions'/(SLUG+'.ipynb'),as_version=4)
for nb,sol in [(student,False),(solution,True)]:
    assert [(c.cell_type,c.source) for c in nb.cells]==[(c.cell_type,c.source) for c in build(sol).cells],('builder drift',sol)
assert sum(c.cell_type=='code' and 'raise NotImplementedError' in c.source for c in student.cells)==4
assert all(c.execution_count is None and not c.outputs for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert '@colab-bootstrap' in next(c.source for c in student.cells if c.cell_type=='code')
assert not any('from relkit.leaderboard import' in c.source for c in student.cells)
payloads=[]
for cell in student.cells:
    assert 'attachment:' not in cell.source
    for token in re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',cell.source):
        b=base64.b64decode(token);Image.open(BytesIO(b)).verify();payloads.append(hashlib.sha256(b).hexdigest())
assert len(payloads)==5
assert set(payloads)=={hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'figures/l056').glob('*.png')}
teacher=json.loads((ROOT/'data/cache/l056/student-audit.json').read_text())
assert teacher==result['summary'],'Live notebook differs from verified audit'
manifest=json.loads((site/'lessons/manifest.json').read_text())
entry=next(x for x in manifest['lessons'] if x['id']==56)
assert entry['slug']==SLUG and entry['labPath']=='labs/'+SLUG+'.ipynb'
assert len({x['id'] for x in manifest['lessons']})==len(manifest['lessons'])
for name in ['index.html','notebooks.html']:
    assert BeautifulSoup((site/name).read_text(),'html.parser').find('meta',attrs={'name':'rdl-manifest-version'})['content']==str(manifest['version'])
stage=Path(tempfile.mkdtemp(prefix='l056-pages-'))
for folder in ['assets','lessons','reference']:shutil.copytree(site/folder,stage/folder)
(stage/'labs/html').mkdir(parents=True)
shutil.copytree(ROOT/'figures/l056',stage/'labs/figures/l056')
shutil.copy2(ROOT/'html'/(SLUG+'.html'),stage/'labs/html'/(SLUG+'.html'))
shutil.copy2(ROOT/(SLUG+'.ipynb'),stage/'labs'/(SLUG+'.ipynb'))
for name in ['index.html','notebooks.html','flashcards.html']:shutil.copy2(site/name,stage/name)
for name in ['l056-reproduction.md','_verify_l056_results.json','_sources_l056.json']:
    assert name in (site/'.github/workflows/pages.yml').read_text()
    shutil.copy2(ROOT/name,stage/'labs'/name)
links=0
for rel in ['lessons/'+SLUG+'.html','reference/tabarena-benchmark-audit.html','labs/html/'+SLUG+'.html']:
    path=stage/rel;soup=BeautifulSoup(path.read_text(),'html.parser')
    for el in soup.find_all(['a','img','script','link']):
        url=el.get('href') or el.get('src')
        if not url:continue
        u=urlsplit(url)
        if u.scheme or u.netloc:continue
        target=(path.parent/unquote(u.path)).resolve() if u.path else path
        assert target.is_relative_to(stage) and target.exists(),(rel,url)
        if u.fragment and target.suffix=='.html':
            assert BeautifulSoup(target.read_text(),'html.parser').find(id=unquote(u.fragment)),(rel,url,'anchor missing')
        links+=1
checks=dict(solution_code_cells=sum(c.cell_type=='code' for c in solution.cells),student_todos=4,
    portable_pngs=5,copied_pages_links=links,staging_path=str(stage),live_notebook_audit='MATCH',
    source_primitive_parity=result['parity'],full_archived_score_analysis='RUN',
    browser='NOT_CHECKED',live_colab='NOT_CHECKED',deployment='NOT_CHECKED',original_training='NOT_RUN',official_elo_table='NOT_REPRODUCED')
(ROOT/'_delivery_l056_results.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
