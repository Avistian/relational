"""Browser checks, notebook visibility and copied Pages delivery."""
import hashlib,json,os,re,subprocess,tempfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0090-gnn-checkpoint'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
errors=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
 page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((ROOT/'lessons'/f'{SLUG}.html').as_uri())
  assert '5.415816' in page.locator('#boundary output').inner_text()
  control=page.locator('#boundary input');control.fill('20');control.dispatch_event('input')
  assert '10.314796' in page.locator('#boundary output').inner_text()
  assert '3.000000' in page.locator('#boundary output').inner_text()
  control.focus();page.keyboard.press('ArrowLeft');assert control.input_value()=='19'
  page.locator('#boundary button').click();assert control.input_value()=='8'
  assert page.locator('#warmup').inner_text().strip()
  assert page.locator('#prediction button').count()>=3
  assert page.locator('#teachback textarea').count()==1
  scroll=page.locator('#boundary .figure-scroll')
  if width==375:
   assert scroll.evaluate('(e)=>e.scrollWidth>e.clientWidth')
   scroll.focus();page.keyboard.press('End');page.wait_for_timeout(100)
   scroll.evaluate('(e)=>e.scrollLeft=e.scrollWidth')
   assert scroll.evaluate('(e)=>e.scrollLeft>0')
   scroll.evaluate('(e)=>e.scrollLeft=0')
  for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Page overflow'
  page.locator('#boundary').screenshot(path=f'/tmp/l090-boundary-{width}.png')
  page.screenshot(path=f'/tmp/l090-page-{width}.png')
 page.goto((LAB/'html'/f'{SLUG}.html').as_uri())
 assert page.locator('img[src^="data:image/png;base64,"]').count()==3
 for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
 nojs=browser.new_page(java_script_enabled=False);nojs.goto((ROOT/'notebooks.html').as_uri());assert nojs.locator('#lab-90 a').count()==4
 browser.close()
assert not errors,errors
nb=nbformat.read(LAB/f'{SLUG}.ipynb',as_version=4);sol=nbformat.read(LAB/'solutions'/f'{SLUG}.ipynb',as_version=4)
assert sum('raise NotImplementedError' in c.source for c in nb.cells if c.cell_type=='code')==5
assert not any('from relkit' in c.source for c in nb.cells if c.cell_type=='code')
assert not any('raise NotImplementedError' in c.source for c in sol.cells if c.cell_type=='code')
assert not any(c.get('outputs') for c in nb.cells if c.cell_type=='code')
assert 'attachment:' not in json.dumps(nb)
assert all(c.execution_count is not None for c in sol.cells if c.cell_type=='code'), 'Teacher solution must be fully executed'
assert json.loads((LAB/'_execution_l090_results.json').read_text())['inline_paper_runs']==100
assert json.loads((LAB/'_clean_environment_l090_results.json').read_text())['paper_runs']==100
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
(LAB/'_delivery_l090_results.json').write_text(json.dumps({'status':'RUNNING'})+'\n')
workflow=(ROOT/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[x[10:] for x in block.splitlines() if x.startswith('          ')]
lines=[x for x in lines if not x.startswith(('VER=','sed -i'))]
with tempfile.TemporaryDirectory(prefix='l090-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=ROOT,check=True,capture_output=True,text=True)
 checked=0
 for path in [stage/'lessons'/f'{SLUG}.html',stage/'reference/gnn-checkpoint.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   dest=(path.parent/unquote(part.path)).resolve();assert dest.exists(),(path,url)
   assert not dest.is_symlink();checked+=1
 for path in ['labs/solutions/'+SLUG+'.ipynb','labs/relkit/checkpoint_l090.py','labs/relkit/gcn_l082.py','labs/_sources_l078.json','labs/_paper_l090_results.json','labs/_run_l090.py','labs/_execution_l090_results.json']:
  assert (stage/path).exists(),path
r={'status':'PASS','browser_widths':[1200,375],'boundary_intervention_and_keyboard':'PASS','portable_figures':3,'student_TODOs':5,'actual_pages_copy_commands':'PASS','copied_local_links':checked,'executed_solution_sha256':hashlib.sha256((LAB/'solutions'/f'{SLUG}.ipynb').read_bytes()).hexdigest(),'page_errors':errors,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(LAB/'_delivery_l090_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
