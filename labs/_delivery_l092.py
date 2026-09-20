"""Browser interactions, notebook completeness and copied Pages link checks."""
import hashlib,json,os,re,subprocess,sys,tempfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0092-meta-paths';preview='--preview' in sys.argv
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
errors=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
 page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((ROOT/'lessons'/f'{SLUG}.html').as_uri())
  assert 'global 0.731; per-node 0.881' in page.locator('#semantic output').inner_text()
  slider=page.locator('#semantic input');slider.fill('4');slider.dispatch_event('input')
  assert 'global 0.953; per-node 0.881' in page.locator('#semantic output').inner_text()
  page.locator('#semantic button').click();assert slider.input_value()=='0'
  slider.focus();page.keyboard.press('ArrowLeft');assert slider.input_value()=='-1'
  assert 'global 0.622; per-node 0.881' in page.locator('#semantic output').inner_text()
  assert page.locator('#warmup').inner_text().strip()
  assert page.locator('#prediction button').count()>=3
  assert page.locator('#teachback textarea').count()==1
  for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Page overflow'
  if width==375:
   scroll=page.locator('.figure-scroll').first;assert scroll.evaluate('(e)=>e.scrollWidth>e.clientWidth')
   scroll.focus();page.keyboard.press('ArrowRight');page.wait_for_timeout(200);assert scroll.evaluate('(e)=>e.scrollLeft>0')
  page.locator('#semantic').screenshot(path=f'/tmp/l092-widget-{width}.png')
  page.screenshot(path=f'/tmp/l092-page-{width}.png')
 page.set_viewport_size({'width':1200,'height':900});page.emulate_media(media='print')
 assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
 page.emulate_media(media='screen');page.goto((LAB/'html'/f'{SLUG}.html').as_uri())
 count=page.locator('img[src^="data:image/png;base64,"]').count();assert count==5,count
 for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
 nojs=browser.new_page(java_script_enabled=False);nojs.goto((ROOT/'notebooks.html').as_uri());assert nojs.locator('#lab-92 a').count()==4
 browser.close()
assert not errors,errors
if preview:print({'preview':'PASS','figures':count,'page_errors':errors});sys.exit()
student=nbformat.read(LAB/f'{SLUG}.ipynb',as_version=4);sol=nbformat.read(LAB/'solutions'/f'{SLUG}.ipynb',as_version=4)
assert sum('raise NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==3
assert not any(c.get('outputs') for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None for c in sol.cells if c.cell_type=='code')
assert 'attachment:' not in json.dumps(student)
assert json.loads((LAB/'_execution_l092_results.json').read_text())['status']=='PASS'
assert json.loads((LAB/'_replay_l092_results.json').read_text())['knn_evaluations']==40
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
(LAB/'_delivery_l092_results.json').write_text('{"status":"RUNNING"}\n')
workflow=(ROOT/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[x[10:] for x in block.splitlines() if x.startswith('          ')]
lines=[x for x in lines if not x.startswith(('VER=','sed -i'))]
with tempfile.TemporaryDirectory(prefix='l092-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=ROOT,check=True,capture_output=True,text=True)
 checked=0
 for path in [stage/'lessons'/f'{SLUG}.html',stage/'reference/meta-paths.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   dest=(path.parent/unquote(part.path)).resolve();assert dest.exists(),(path,url)
   assert not dest.is_symlink();checked+=1
 for name in ['labs/solutions/'+SLUG+'.ipynb','labs/relkit/han_l092.py','labs/_sources_l092.json','labs/_paper_l092_results.json','labs/_run_l092.py','labs/sources/han-l092/layers.py']:
  assert (stage/name).exists(),name
result={'status':'PASS','browser_widths':[1200,375],'semantic_intervention_and_keyboard':'PASS','portable_figures':5,'student_TODOs':3,'print_media_layout':'PASS','actual_pages_copy_commands':'PASS','copied_local_links':checked,'saved_solution_scope':'diagnostic outputs; independent full replay in _replay_l092_results.json','executed_solution_sha256':hashlib.sha256((LAB/'solutions'/f'{SLUG}.ipynb').read_bytes()).hexdigest(),'page_errors':errors,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(LAB/'_delivery_l092_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
