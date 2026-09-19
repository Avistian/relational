"""Audit portable notebook source and links in the actual copied Pages output."""
import ast,base64,json,re,subprocess,tempfile,hashlib
from pathlib import Path
from urllib.parse import urlsplit,unquote
import nbformat
from bs4 import BeautifulSoup
from _build_l080 import CHECKS
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0080-year-2-exit-exam'
student=nbformat.read(LAB/f'{SLUG}.ipynb',as_version=4);teacher=nbformat.read(LAB/'solutions'/f'{SLUG}.ipynb',as_version=4)
assert sum('raise NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==3
assert all(not c.outputs and c.execution_count is None for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and all(o.output_type!='error' for o in c.outputs) for c in teacher.cells if c.cell_type=='code')
for nb in [student,teacher]:
    text='\n'.join(c.source for c in nb.cells);images=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',text)
    assert len(images)==2 and all(base64.b64decode(x).startswith(b'\x89PNG') for x in images)
    assert 'attachment:' not in text
# Compare every visible architecture/driver definition with canonical source.
expected={}
for name in ['ft_l080.py','tabm_v2.py','tabpfn_l064_v2.py','exit_l080.py']:
    for n in ast.parse((LAB/'relkit'/name).read_text()).body:
        if isinstance(n,(ast.ClassDef,ast.FunctionDef)):
            if name=='tabpfn_l064_v2.py' and n.name in ['load_dataset','run_experiment']:continue
            expected[n.name]=ast.dump(n,include_attributes=False)
actual={};env={}
for c in teacher.cells:
    if c.cell_type!='code' or '@colab-bootstrap' in c.source:continue
    for n in ast.parse(c.source).body:
        if isinstance(n,(ast.ClassDef,ast.FunctionDef)):actual[n.name]=ast.dump(n,include_attributes=False)
assert actual==expected,'Visible source drift'
for name in ['_check_l080_results.json','_execution_l080_results.json','_browser_l080_results.json']:
    assert json.loads((LAB/name).read_text())['status']=='PASS'
workflow=(ROOT/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - ',1)[0]
lines=[ln[10:] for ln in block.splitlines()];lines=[ln for ln in lines if not ln.startswith('VER=') and not ln.startswith('sed -i')];checked=0
with tempfile.TemporaryDirectory(prefix='l080-pages-') as tmp:
    stage=Path(tmp)/'public';script='\n'.join(lines).replace('public/',str(stage)+'/').replace('mkdir -p public',f'mkdir -p {stage}')
    subprocess.run(['bash','-e','-c',script],cwd=ROOT,check=True,capture_output=True)
    for relative in [f'lessons/{SLUG}.html',f'reference/{SLUG}.html',f'labs/html/{SLUG}.html']:
        page=stage/relative;soup=BeautifulSoup(page.read_text(),'html.parser')
        for tag in soup.find_all(['a','img','script','link']):
            raw=tag.get('href') or tag.get('src')
            if not raw:continue
            u=urlsplit(raw)
            if u.scheme or not u.path:continue
            target=(page.parent/unquote(u.path)).resolve()
            assert target.exists(),str(target)
            checked+=1
report={'status':'PASS','copied_pages_links':checked,'inline_source_definitions':len(actual),'student_todos':3,'portable_figures':2,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(LAB/'_delivery_l080_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
