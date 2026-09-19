"""Artifact coherence, source provenance and real copied-Pages link validation."""
import ast,base64,hashlib,json,re,subprocess,tempfile
from pathlib import Path
from urllib.parse import urlsplit,unquote
import nbformat
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0075-pytorch-frame-row-encoder'
def run():
 student=nbformat.read(LAB/f'{SLUG}.ipynb',as_version=4);teacher=nbformat.read(LAB/'solutions'/f'{SLUG}.ipynb',as_version=4)
 assert sum('raise NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==3
 assert all(not c.outputs and c.execution_count is None for c in student.cells if c.cell_type=='code')
 assert all(c.execution_count is not None and all(o.output_type!='error' for o in c.outputs) for c in teacher.cells if c.cell_type=='code')
 for nb in [student,teacher]:
  images=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)','\n'.join(c.source for c in nb.cells))
  assert len(images)==4 and all(base64.b64decode(s).startswith(b'\x89PNG') for s in images)
 expected={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((LAB/'relkit/frame_l075.py').read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 actual={}
 for c in teacher.cells:
  if c.cell_type!='code' or '@colab-bootstrap' in c.source:continue
  for n in ast.parse(c.source).body:
   if isinstance(n,(ast.FunctionDef,ast.ClassDef)):actual[n.name]=ast.dump(n,include_attributes=False)
 assert all(actual[k]==v for k,v in expected.items()),'Notebook definitions drifted'
 digest=hashlib.sha256((LAB/'relkit/frame_l075.py').read_bytes()).hexdigest()
 for name in ['_execution_l075_results.json','_verify_l075_results.json']:
  r=json.loads((LAB/name).read_text());assert r['status']=='PASS' and r['implementation_sha256']==digest
 assert json.loads((LAB/'_browser_l075_results.json').read_text())['status']=='PASS'
 r=json.loads((LAB/'_verify_l075_results.json').read_text())
 assert hashlib.sha256((LAB/'data/cache/credit_g.parquet').read_bytes()).hexdigest()==r['data_sha256']
 assert not set(r['real_table']['train_ids'])&set(r['real_table']['query_ids'])
 for name,record in json.loads((LAB/'_sources_l075.json').read_text())['files'].items():
  assert hashlib.sha256((LAB/name).read_bytes()).hexdigest()==record['sha256']
 # Copy exactly the Pages build block (skip only cache-busting that needs GitHub env).
 workflow=(ROOT/'.github/workflows/pages.yml').read_text()
 block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - ',1)[0]
 lines=[ln[10:] for ln in block.splitlines()];lines=[ln for ln in lines if not ln.startswith('VER=') and not ln.startswith('sed -i')]
 checked=0
 with tempfile.TemporaryDirectory(prefix='l075-pages-') as tmp:
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
     target_soup=BeautifulSoup(target.read_text(),'html.parser')
     assert target_soup.find(id=u.fragment) or target_soup.find(id=unquote(u.fragment)),f'Missing fragment: {raw}'
    checked+=1
 manifest=json.loads((ROOT/'lessons/manifest.json').read_text())
 assert next(z for z in manifest['lessons'] if z['id']==75)['labPath']==f'labs/{SLUG}.ipynb'
 assert f'lessons/{SLUG}.html' in (ROOT/'notebooks.html').read_text()
 report={'status':'PASS','student_todos':3,'portable_figures_per_notebook':4,'visible_canonical_definitions':len(expected),'copied_pages_links':checked,'source_hashes':'PASS','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
 (LAB/'_delivery_l075_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':run()
