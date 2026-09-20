"""Check real browser behavior and a copied Pages tree, with one browser at a time."""
import hashlib,json,os,re,subprocess,tempfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0094-hin-survey'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
errors=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
 page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri())
  page.screenshot(path=f'/tmp/l094-top-{width}.png')
  assert page.locator('#warmup').inner_text().strip()
  select=page.locator('#route-widget select');output=page.locator('#route-widget output');check=page.locator('#route-widget input')
  assert 'Ada–Bo: 1; Ada–Cy: 0' in output.inner_text()
  for route,baseline,changed in [('paper','1; Ada–Cy: 0','0; Ada–Cy: 0'),('venue','2; Ada–Cy: 1','1; Ada–Cy: 1'),('both','1; Ada–Cy: 0','0; Ada–Cy: 0')]:
   select.select_option(route);check.uncheck();assert baseline in output.inner_text();check.check();assert changed in output.inner_text()
  page.locator('#route-widget button').click();assert select.input_value()=='paper' and not check.is_checked()
  select.focus();page.keyboard.press('ArrowDown');assert select.input_value()=='venue'
  assert page.locator('#prediction button').count()>=3 and page.locator('#teachback textarea').count()==1
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'page overflow'
  for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
  page.screenshot(path=f'/tmp/l094-page-{width}.png');page.locator('#route-widget').screenshot(path=f'/tmp/l094-widget-{width}.png')
  if width==375:
   sc=page.locator('.figure-scroll').first;sc.focus();page.keyboard.press('ArrowRight');page.wait_for_timeout(150);assert sc.evaluate('(e)=>e.scrollLeft>0')
 page.set_viewport_size({'width':1200,'height':900});page.emulate_media(media='print');assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1');page.emulate_media(media='screen')
 page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('img[src^="data:image/png;base64,"]').count()==2
 for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
 page.locator('img').first.screenshot(path='/tmp/l094-notebook-figure.png')
 nojs=browser.new_page(java_script_enabled=False);nojs.goto((R/'notebooks.html').as_uri());assert nojs.locator('#lab-94 a').count()==4
 browser.close()
assert not errors,errors
student=nbformat.read(P/f'{S}.ipynb',as_version=4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
assert sum('raise NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==3
assert not any(c.get('outputs') for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None for c in solution.cells if c.cell_type=='code')
assert 'attachment:' not in json.dumps(student)
assert [c.source for c in student.cells if c.cell_type=='code' and not c.metadata.get('task')]==[c.source for c in solution.cells if c.cell_type=='code' and not c.metadata.get('task')]
assert json.loads((P/'_execution_l094_results.json').read_text())['graph_count_parity']=='EXACT'
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
(P/'_delivery_l094_results.json').write_text('{"status":"RUNNING"}\n')
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0];lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))]
with tempfile.TemporaryDirectory(prefix='l094-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 checked=0
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/hin-taxonomy.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   dest=(path.parent/unquote(part.path)).resolve();assert dest.exists(),(path,url);assert not dest.is_symlink();checked+=1
 for relative in ['labs/sources/hin-l094/survey.pdf','labs/sources/hin-l094/table1.json','labs/sources/hin-l094/prior-evidence.json','labs/relkit/hin_l094.py','labs/relkit/oag_read_l094.py','labs/_run_l094.py','labs/l094-reproduction.md','labs/solutions/'+S+'.ipynb']:
  assert (stage/relative).exists(),relative
result={'status':'PASS','browser_widths':[1200,375],'route_states_checked':6,'keyboard_reset_print':'PASS','portable_figures':2,'solution_code_cells':16,'copied_pages_local_links':checked,'javascript_errors':errors,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/'_delivery_l094_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
