"""Check actual inline execution, source identity, and links inside a copied Pages build."""
import ast,base64,hashlib,json,re,subprocess,tempfile
from pathlib import Path
from urllib.parse import urlsplit,unquote
import nbformat
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0083-graphsage'
student=nbformat.read(LAB/f'{SLUG}.ipynb',as_version=4);teacher=nbformat.read(LAB/'solutions'/f'{SLUG}.ipynb',as_version=4)
assert sum('raise NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==3
assert all(not c.outputs and c.execution_count is None for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and all(o.output_type!='error' for o in c.outputs) for c in teacher.cells if c.cell_type=='code')
for nb in [student,teacher]:
 text='\n'.join(c.source for c in nb.cells);images=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',text)
 assert len(images)==4 and all(base64.b64decode(x).startswith(b'\x89PNG') for x in images)
 assert 'attachment:' not in text
expected={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((LAB/'relkit/graphsage_l083.py').read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
actual={}
for c in teacher.cells:
 if c.cell_type=='code':
  for n in ast.parse(c.source).body:
   if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in expected:actual[n.name]=ast.dump(n,include_attributes=False)
assert actual==expected
for name in ['_verify_l083_results.json','_execution_l083_results.json','_browser_l083_results.json']:
 assert json.loads((LAB/name).read_text())['status']=='PASS'
r=json.loads((LAB/'_paper_l083_results.json').read_text());replay=json.loads((LAB/'l083-student-results.json').read_text())
assert [v['seed'] for v in r['runs']]==[123,124,125]
for run,other in zip(r['runs'],replay['runs']):
 assert run['selected_lr']==max(run['candidates'],key=lambda c:c['validation']['micro_f1'])['lr']
 assert len(run['candidates'])==3 and all(len(c['trace'])==10 for c in run['candidates'])
 for record in [run,other]:
  for c in record['candidates']:c.pop('seconds')
 assert run==other
assert r['implementation_sha256']==hashlib.sha256((LAB/'relkit/graphsage_l083.py').read_bytes()).hexdigest()
assert r['source_manifest_sha256']==hashlib.sha256((LAB/'_sources_l083.json').read_bytes()).hexdigest()
execution=json.loads((LAB/'_execution_l083_results.json').read_text())
assert execution['notebook_sha256']==hashlib.sha256((LAB/'solutions'/f'{SLUG}.ipynb').read_bytes()).hexdigest()
assert abs(sum(v['test']['micro_f1'] for v in r['runs'])/3-r['mean'])<1e-12
workflow=(ROOT/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - ',1)[0]
lines=[ln[10:] for ln in block.splitlines()];lines=[ln for ln in lines if not ln.startswith('VER=') and not ln.startswith('sed -i')];checked=0
with tempfile.TemporaryDirectory(prefix='l083-pages-') as tmp:
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
    dest=BeautifulSoup(target.read_text(),'html.parser');assert dest.find(id=u.fragment) or dest.find(id=unquote(u.fragment)),(target,u.fragment)
   checked+=1
 assert (stage/'labs/relkit/graphsage_l083.py').is_file()
 assert (stage/'labs/solutions'/f'{SLUG}.ipynb').is_file()
report={'status':'PASS','copied_pages_links':checked,'inline_definitions':len(actual),'student_todos':3,'portable_figures':4,'full_ppi_searches':3,'mean':r['mean'],'sample_sd':r['sample_sd'],'historical_parity':'INCOMPARABLE','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(LAB/'_delivery_l083_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
