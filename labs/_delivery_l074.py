"""Canonical code, metric reconstruction, portable assets and copied Pages links."""
import ast,base64,hashlib,json,subprocess,tempfile,re
from pathlib import Path
from urllib.parse import unquote,urlsplit
import nbformat
from bs4 import BeautifulSoup
from sklearn.metrics import r2_score
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0074-carte-cross-table-transfer'
def run():
    student=nbformat.read(LAB/f'{SLUG}.ipynb',as_version=4);teacher=nbformat.read(LAB/'solutions'/f'{SLUG}.ipynb',as_version=4)
    assert sum('raise NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==3
    assert all(not c.outputs and c.execution_count is None for c in student.cells if c.cell_type=='code')
    assert all(c.execution_count is not None and all(o.output_type!='error' for o in c.outputs) for c in teacher.cells if c.cell_type=='code')
    for nb in [student,teacher]:
        images=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)','\n'.join(c.source for c in nb.cells))
        assert len(images)==5 and all(base64.b64decode(s).startswith(b'\x89PNG') for s in images)
    expected={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((LAB/'relkit/carte_l074.py').read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    actual={}
    for c in teacher.cells:
        if c.cell_type!='code' or '@colab-bootstrap' in c.source:continue
        for n in ast.parse(c.source).body:
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)):actual[n.name]=ast.dump(n,include_attributes=False)
    assert all(actual[k]==v for k,v in expected.items())
    execution=json.loads((LAB/'_execution_l074_results.json').read_text());assert execution['status']=='PASS'
    assert execution['implementation_sha256']==hashlib.sha256((LAB/'relkit/carte_l074.py').read_bytes()).hexdigest()
    r=json.loads((LAB/'_verify_l074_results.json').read_text());assert len(r['records'])==54
    assert len({(z['dataset'],z['seed'],z['arm']) for z in r['records']})==54
    for p,h in r['source_hashes'].items():assert hashlib.sha256((LAB/p).read_bytes()).hexdigest()==h
    for z in r['records']:assert abs(r2_score(z['target'],z['prediction'])-z['r2'])<1e-12
    for dataset in r['config']['datasets']:
        for seed in r['config']['seeds']:
            rows=[z for z in r['records'] if z['dataset']==dataset and z['seed']==seed]
            assert len(rows)==6 and all(z['target']==rows[0]['target'] for z in rows)
    for s in r['splits']:
        tr,va,te=map(set,[s['train'],s['validation'],s['test']]);assert not tr&va and not tr&te and not va&te
        assert [len(tr),len(va),len(te)]==[64,64,256]
    manifest=json.loads((LAB/'data/l074/manifest.json').read_text())
    for f,h in manifest['files'].items():assert hashlib.sha256((LAB/'data/l074'/f).read_bytes()).hexdigest()==h
    provenance=json.loads((LAB/'_sources_l074.json').read_text())
    for f,h in provenance['files'].items():assert hashlib.sha256((LAB/f).read_bytes()).hexdigest()==h
    workflow=(ROOT/'.github/workflows/pages.yml').read_text()
    block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - ',1)[0]
    lines=[ln[10:] for ln in block.splitlines()];lines=[ln for ln in lines if not ln.startswith('VER=') and not ln.startswith('sed -i')]
    checked=0
    with tempfile.TemporaryDirectory(prefix='l074-pages-') as tmp:
        stage=Path(tmp)/'public';script='\n'.join(lines).replace('public/',str(stage)+'/').replace('mkdir -p public',f'mkdir -p {stage}')
        subprocess.run(['bash','-e','-c',script],cwd=ROOT,check=True,capture_output=True)
        for relative in [f'lessons/{SLUG}.html',f'reference/{SLUG}.html',f'labs/html/{SLUG}.html']:
            page=stage/relative;soup=BeautifulSoup(page.read_text(),'html.parser')
            for tag in soup.find_all(['a','img','link','script']):
                u=tag.get('href') or tag.get('src') or '';parts=urlsplit(u)
                if not u or parts.scheme or u.startswith('#'):continue
                target=(page.parent/unquote(parts.path)).resolve();assert target.exists(),(relative,u)
                if parts.fragment and target.suffix=='.html':assert BeautifulSoup(target.read_text(),'html.parser').find(id=unquote(parts.fragment)),u
                checked+=1
        for f in manifest['files']:assert hashlib.sha256((stage/'labs/data/l074'/f).read_bytes()).hexdigest()==manifest['files'][f]
    assert json.loads((LAB/'_browser_l074_results.json').read_text())['status']=='PASS'
    assert json.loads((LAB/'_check_l074_results.json').read_text())['status']=='PASS'
    assert json.loads((LAB/'_numeric_l074_results.json').read_text())['status']=='PASS'
    report={'status':'PASS','student_todos':3,'portable_figures':5,'canonical_definition_parity':True,'evaluations':54,'copied_pages_links':checked,'solution_execution':execution,'browser':'PASS','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED','larger_run':'NOT_RUN'}
    (LAB/'_delivery_l074_results.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':run()
