"""Check source alignment, source pins, notebook evidence and copied Pages links."""
import ast,base64,hashlib,json,re,subprocess,tempfile
from pathlib import Path
from urllib.parse import urlsplit,unquote
import nbformat
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0082-gcn'
student=nbformat.read(LAB/f'{SLUG}.ipynb',as_version=4);teacher=nbformat.read(LAB/'solutions'/f'{SLUG}.ipynb',as_version=4)
assert sum('raise NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==3
assert all(not c.outputs and c.execution_count is None for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and all(o.output_type!='error' for o in c.outputs) for c in teacher.cells if c.cell_type=='code')
for nb in [student,teacher]:
    text='\n'.join(c.source for c in nb.cells);images=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',text)
    assert len(images)==4 and all(base64.b64decode(x).startswith(b'\x89PNG') for x in images)
    assert 'attachment:' not in text
expected={}
for name in ['gcn_l082.py']:
    for n in ast.parse((LAB/'relkit'/name).read_text()).body:
        if isinstance(n,(ast.ClassDef,ast.FunctionDef)):expected[n.name]=ast.dump(n,include_attributes=False)
actual={}
for c in teacher.cells:
    if c.cell_type!='code':continue
    for n in ast.parse(c.source).body:
        if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in expected:actual[n.name]=ast.dump(n,include_attributes=False)
assert actual==expected,'Canonical implementation differs from notebook'
provenance=json.loads((LAB/'_sources_l082.json').read_text())
for name,digest in provenance['local_sha256'].items():assert hashlib.sha256((LAB/name).read_bytes()).hexdigest()==digest
for record in provenance['files']:assert hashlib.sha256((LAB/record['path']).read_bytes()).hexdigest()==record['sha256']
for name in ['_verify_l082_results.json','_execution_l082_results.json','_browser_l082_results.json']:
    assert json.loads((LAB/name).read_text())['status']=='PASS'
r=json.loads((LAB/'_paper_l082_results.json').read_text())
replay=json.loads((LAB/'l082-student-results.json').read_text())
assert r['runs']==replay['runs']
assert [v['seed'] for v in r['runs']]==list(range(100))
assert all(v['epochs']==len(v['validation_loss']) and 1<=v['epochs']<=200 for v in r['runs'])
assert r['implementation_sha256']==provenance['local_sha256']['relkit/gcn_l082.py']
assert r['source_manifest_sha256']==hashlib.sha256((LAB/'_sources_l078.json').read_bytes()).hexdigest()
assert abs(sum(v['test_accuracy'] for v in r['runs'])/100-r['mean'])<1e-12
assert json.loads((LAB/'_isolation_l082_results.json').read_text())['status']=='PASS'
workflow=(ROOT/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - ',1)[0]
lines=[ln[10:] for ln in block.splitlines()];lines=[ln for ln in lines if not ln.startswith('VER=') and not ln.startswith('sed -i')];checked=0
with tempfile.TemporaryDirectory(prefix='l082-pages-') as tmp:
    stage=Path(tmp)/'public';script='\n'.join(lines).replace('public/',str(stage)+'/').replace('mkdir -p public',f'mkdir -p {stage}')
    subprocess.run(['bash','-e','-c',script],cwd=ROOT,check=True,capture_output=True)
    for relative in [f'lessons/{SLUG}.html',f'reference/{SLUG}.html',f'labs/html/{SLUG}.html']:
        page=stage/relative;soup=BeautifulSoup(page.read_text(),'html.parser')
        for tag in soup.find_all(['a','img','script','link']):
            raw=tag.get('href') or tag.get('src')
            if not raw:continue
            u=urlsplit(raw)
            if u.scheme:continue
            target=(page.parent/unquote(u.path)).resolve() if u.path else page
            assert target.exists(),str(target)
            if u.fragment and target.suffix=='.html':
                dest=BeautifulSoup(target.read_text(),'html.parser')
                assert dest.find(id=u.fragment) or dest.find(id=unquote(u.fragment)),(str(target),u.fragment)
            checked+=1
    for name in ['gcn_l082.py']:assert (stage/'labs/relkit'/name).is_file()
report={'status':'PASS','copied_pages_links':checked,'inline_source_definitions':len(actual),'student_todos':3,'portable_figures':4,'live_colab':'NOT_CHECKED','remote_execution':'NOT_RUN','full_cora_port':'100 runs; PASS','original_tensorflow_parity':'NOT_RUN','deployment':'NOT_CHECKED'}
(LAB/'_delivery_l082_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
