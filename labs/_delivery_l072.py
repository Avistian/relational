"""Validate notebooks, evidence, and local links in a copied Pages staging tree."""
import ast,base64,hashlib,json,re,shutil,subprocess,tempfile
from pathlib import Path
from urllib.parse import unquote,urlsplit
import nbformat
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0072-scarf-subtab-contrastive-views'
def run():
    student=nbformat.read(LAB/f'{SLUG}.ipynb',as_version=4)
    teacher=nbformat.read(LAB/'solutions'/f'{SLUG}.ipynb',as_version=4)
    blanks=[c for c in student.cells if c.cell_type=='code' and 'raise NotImplementedError' in c.source]
    assert len(blanks)==3
    assert all(not c.outputs and c.execution_count is None for c in student.cells if c.cell_type=='code')
    for nb in [student,teacher]:
        images=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)','\n'.join(c.source for c in nb.cells))
        assert len(images)==4
        assert all(base64.b64decode(s).startswith(b'\x89PNG') for s in images)
    expected=ast.parse((LAB/'relkit/contrastive_l072.py').read_text())
    expected_nodes={n.name:ast.dump(n,include_attributes=False) for n in expected.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    actual_nodes={}
    for c in teacher.cells:
        if c.cell_type!='code' or '@colab-bootstrap' in c.source or c.source.startswith('%%writefile '):continue
        for n in ast.parse(c.source).body:
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)):actual_nodes[n.name]=ast.dump(n,include_attributes=False)
    assert all(actual_nodes[k]==v for k,v in expected_nodes.items())
    execution=json.loads((LAB/'_execution_l072_results.json').read_text());assert execution['status']=='PASS'
    assert execution['implementation_sha256']==hashlib.sha256((LAB/'relkit/contrastive_l072.py').read_bytes()).hexdigest()
    r=json.loads((LAB/'_verify_l072_results.json').read_text());assert len(r['records'])==54
    for z in r['records']:assert sum(a==b for a,b in zip(z['prediction'],z['target']))/len(z['target'])==z['accuracy']
    for s in r['splits']:
        tr,va,te=map(set,[s['train'],s['validation'],s['test']]);assert not tr&va and not tr&te and not va&te and set(s['labeled'])<=tr
    # Execute the actual workflow's copy commands, not symlinks, in a temporary root.
    workflow=(ROOT/'.github/workflows/pages.yml').read_text()
    block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - ',1)[0]
    lines=[ln[10:] for ln in block.splitlines()]
    # Version extraction/sed modifies source pages; stage copies already carry metadata.
    lines=[ln for ln in lines if not ln.startswith('VER=') and not ln.startswith('sed -i')]
    checked=0
    with tempfile.TemporaryDirectory(prefix='l072-pages-') as tmp:
        stage=Path(tmp)/'public'
        script='\n'.join(lines).replace('public/',str(stage)+'/').replace('mkdir -p public',f'mkdir -p {stage}')
        subprocess.run(['bash','-e','-c',script],cwd=ROOT,check=True,capture_output=True)
        for relative in [f'lessons/{SLUG}.html',f'reference/{SLUG}.html',f'labs/html/{SLUG}.html']:
            page=stage/relative;soup=BeautifulSoup(page.read_text(),'html.parser')
            for tag in soup.find_all(['a','img','link','script']):
                u=tag.get('href') or tag.get('src') or '';parts=urlsplit(u)
                if not u or parts.scheme or u.startswith('#'):continue
                target=(page.parent/unquote(parts.path)).resolve()
                assert target.exists(),(relative,u)
                if parts.fragment and target.suffix=='.html':
                    other=BeautifulSoup(target.read_text(),'html.parser');assert other.find(id=unquote(parts.fragment)),(u,'missing fragment')
                checked+=1
    browser=json.loads((LAB/'_browser_l072_results.json').read_text());assert browser['status']=='PASS'
    report={'status':'PASS','student_todos':3,'portable_figures':4,'canonical_definition_parity':True,'evaluations':54,'copied_pages_links':checked,'solution_execution':execution,'browser':'PASS','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED','larger_run':'NOT_RUN'}
    (LAB/'_delivery_l072_results.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':run()
