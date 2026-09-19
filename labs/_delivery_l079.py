"""Check real failure cases, notebook parity and the actual copied Pages workflow."""
import ast,base64,copy,gzip,hashlib,json,re,subprocess,tempfile
from pathlib import Path
from urllib.parse import urlsplit,unquote
import nbformat
from bs4 import BeautifulSoup
from relkit.comparison_l060 import audit_records
from _build_l079 import CHECKS
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0079-neural-tabular-decision-guide'
student=nbformat.read(LAB/f'{SLUG}.ipynb',as_version=4);teacher=nbformat.read(LAB/'solutions'/f'{SLUG}.ipynb',as_version=4)
assert sum('raise NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==2
assert all(not c.outputs and c.execution_count is None for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and all(o.output_type!='error' for o in c.outputs) for c in teacher.cells if c.cell_type=='code')
for nb in [student,teacher]:
    text='\n'.join(c.source for c in nb.cells);images=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',text)
    assert len(images)==3 and all(base64.b64decode(x).startswith(b'\x89PNG') for x in images)
    assert 'attachment:' not in text
expected={}
for module in ['comparison_l060.py','decision_guide.py']:
    for n in ast.parse((LAB/'relkit'/module).read_text()).body:
        if isinstance(n,ast.FunctionDef):expected[n.name]=ast.dump(n,include_attributes=False)
actual={};env={}
for c in teacher.cells:
    if c.cell_type!='code' or '@colab-bootstrap' in c.source:continue
    nodes=[n for n in ast.parse(c.source).body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<inline>','exec'),env)
    for n in nodes:
        if isinstance(n,ast.FunctionDef):actual[n.name]=ast.dump(n,include_attributes=False)
assert actual==expected,'Inline source drift'
for code in CHECKS.values():exec(code,env)
for name,code in CHECKS.items():
    saved=env[name];env[name]=lambda *a:None
    try:exec(code,env)
    except (AssertionError,TypeError):pass
    else:raise AssertionError('CHECK failed to detect broken '+name)
    finally:env[name]=saved
original=json.loads(gzip.decompress((LAB/'data/l079/l060-v2.json.gz').read_bytes()));mutations=[]
for kind in ['missing_record','metric','selection','test_order','split_overlap']:
    r=copy.deepcopy(original)
    if kind=='missing_record':r['records'].pop()
    if kind=='metric':r['records'][0]['error']+=.1
    if kind=='selection':r['records'][0]['selected']=1-r['records'][0]['selected']
    if kind=='test_order':r['records'][0]['test_ids'].reverse()
    if kind=='split_overlap':
        ids=next(iter(r['datasets'].values()))['ids'];ids['test'][0]=ids['train'][0]
    try:audit_records(r)
    except ValueError:mutations.append(kind)
    else:raise AssertionError('Audit accepted '+kind)
for name in ['_verify_l079_results.json','_execution_l079_results.json','_browser_l079_results.json']:assert json.loads((LAB/name).read_text())['status']=='PASS'
workflow=(ROOT/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - ',1)[0]
lines=[ln[10:] for ln in block.splitlines()];lines=[ln for ln in lines if not ln.startswith('VER=') and not ln.startswith('sed -i')];checked=0
with tempfile.TemporaryDirectory(prefix='l079-pages-') as tmp:
    stage=Path(tmp)/'public';script='\n'.join(lines).replace('public/',str(stage)+'/').replace('mkdir -p public',f'mkdir -p {stage}')
    subprocess.run(['bash','-e','-c',script],cwd=ROOT,check=True,capture_output=True)
    for relative in [f'lessons/{SLUG}.html',f'reference/{SLUG}.html',f'labs/html/{SLUG}.html']:
        page=stage/relative;soup=BeautifulSoup(page.read_text(),'html.parser')
        for tag in soup.find_all(['a','img','script','link']):
            raw=tag.get('href') or tag.get('src')
            if not raw:continue
            u=urlsplit(raw)
            if u.scheme or u.netloc:continue
            target=(page.parent/unquote(u.path)).resolve() if u.path else page
            assert target.exists(),f'Broken local link {relative}: {raw}'
            if u.fragment and target.suffix=='.html':
                other=BeautifulSoup(target.read_text(),'html.parser');assert other.find(id=u.fragment) or other.find(id=unquote(u.fragment)),raw
            checked+=1
    for file in ['labs/data/l079/l060-v2.json.gz','labs/_sources_l079.json','solutions/l079-example-guide.md']:assert (stage/file).exists()
manifest=json.loads((ROOT/'lessons/manifest.json').read_text());assert next(x for x in manifest['lessons'] if x['id']==79)['labPath']==f'labs/{SLUG}.ipynb'
r={'status':'PASS','student_todos':2,'inline_functions':len(expected),'portable_figures_per_notebook':3,'corruptions_rejected':mutations,'copied_pages_links':checked,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(LAB/'_delivery_l079_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
