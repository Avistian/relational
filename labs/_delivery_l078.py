"""Notebook/source parity, mutation-sensitive tasks and copied Pages delivery."""
import ast,base64,hashlib,json,re,subprocess,tempfile
from pathlib import Path
from urllib.parse import urlsplit,unquote
import nbformat
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0078-message-passing-preview'
def run():
    student=nbformat.read(LAB/f'{SLUG}.ipynb',as_version=4);teacher=nbformat.read(LAB/'solutions'/f'{SLUG}.ipynb',as_version=4)
    assert sum('raise NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==3
    assert all(not c.outputs and c.execution_count is None for c in student.cells if c.cell_type=='code')
    assert all(c.execution_count is not None and all(o.output_type!='error' for o in c.outputs) for c in teacher.cells if c.cell_type=='code')
    for nb in [student,teacher]:
        text='\n'.join(c.source for c in nb.cells);images=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',text)
        assert len(images)==4 and all(base64.b64decode(x).startswith(b'\x89PNG') for x in images)
        assert 'attachment:' not in text
    source=(LAB/'relkit/message_passing.py').read_text()
    expected={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    actual={};env={}
    for c in teacher.cells:
        if c.cell_type!='code' or '@colab-bootstrap' in c.source:continue
        tree=ast.parse(c.source)
        nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef))]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<inline>','exec'),env)
        for n in nodes:
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)):actual[n.name]=ast.dump(n,include_attributes=False)
    assert all(actual[k]==v for k,v in expected.items()),'Inline implementation drift'
    from _build_l078 import CHECKS
    for check in CHECKS.values():exec(check,env)
    mutations=[]
    for name in CHECKS:
        saved=env[name];env[name]=lambda *args:env['np'].zeros_like(args[0],dtype=float)
        try:exec(CHECKS[name],env)
        except (AssertionError,ValueError):mutations.append(name)
        else:raise AssertionError(name+' CHECK did not detect a broken implementation')
        finally:env[name]=saved
    digest=hashlib.sha256(source.encode()).hexdigest();r=json.loads((LAB/'_paper_l078_results.json').read_text())
    assert r['implementation_sha256']==digest and len(r['runs'])==100
    assert r['source_manifest_sha256']==hashlib.sha256((LAB/'_sources_l078.json').read_bytes()).hexdigest()
    for name in ['_execution_l078_results.json','_verify_l078_results.json','_browser_l078_results.json']:assert json.loads((LAB/name).read_text())['status']=='PASS'
    workflow=(ROOT/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - ',1)[0]
    lines=[ln[10:] for ln in block.splitlines()];lines=[ln for ln in lines if not ln.startswith('VER=') and not ln.startswith('sed -i')];checked=0
    with tempfile.TemporaryDirectory(prefix='l078-pages-') as tmp:
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
    manifest=json.loads((ROOT/'lessons/manifest.json').read_text());assert next(x for x in manifest['lessons'] if x['id']==78)['labPath']==f'labs/{SLUG}.ipynb'
    result={'status':'PASS','student_todos':3,'portable_figures_per_notebook':4,'visible_canonical_definitions':len(expected),'mutations_detected':mutations,'copied_pages_links':checked,'paper_port_runs':100,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
    (LAB/'_delivery_l078_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':run()
