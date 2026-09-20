"""Browser behavior, notebook visibility and actual copied Pages staging."""
import hashlib,json,os,re,subprocess,tempfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0091-r-gcn'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
errors=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
 page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((ROOT/'lessons'/f'{SLUG}.html').as_uri())
  assert 'ReLU(3) = 3' in page.locator('#typed output').inner_text()
  select=page.locator('#typed select');select.select_option('returns')
  assert 'ReLU(12) = 12' in page.locator('#typed output').inner_text()
  page.locator('#typed button').click();assert select.input_value()=='buys'
  select.focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert select.input_value()=='returns'
  assert page.locator('#warmup').inner_text().strip()
  assert page.locator('#prediction button').count()>=3
  assert page.locator('#teachback textarea').count()==1
  for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Page overflow'
  if width==375:
   scroll=page.locator('#typed .figure-scroll');assert scroll.evaluate('(e)=>e.scrollWidth>e.clientWidth')
   scroll.evaluate('(e)=>e.scrollLeft=e.scrollWidth');assert scroll.evaluate('(e)=>e.scrollLeft>0')
   scroll.evaluate('(e)=>e.scrollLeft=0')
  page.locator('#typed').screenshot(path=f'/tmp/l091-widget-{width}.png')
  page.screenshot(path=f'/tmp/l091-page-{width}.png')
 page.set_viewport_size({'width':1200,'height':900});page.emulate_media(media='print')
 assert page.locator('img').count()==4
 page.pdf(path='/tmp/l091-print.pdf',format='A4',print_background=True)
 page.emulate_media(media='screen');page.goto((LAB/'html'/f'{SLUG}.html').as_uri())
 assert page.locator('img[src^="data:image/png;base64,"]').count()==4
 for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
 nojs=browser.new_page(java_script_enabled=False);nojs.goto((ROOT/'notebooks.html').as_uri());assert nojs.locator('#lab-91 a').count()==4
 browser.close()
assert not errors,errors
student=nbformat.read(LAB/f'{SLUG}.ipynb',as_version=4);sol=nbformat.read(LAB/'solutions'/f'{SLUG}.ipynb',as_version=4)
assert sum('raise NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==3
assert not any('from relkit' in c.source for c in student.cells if c.cell_type=='code')
assert not any(c.get('outputs') for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None for c in sol.cells if c.cell_type=='code')
assert 'attachment:' not in json.dumps(student)
assert json.loads((LAB/'_execution_l091_results.json').read_text())['inline_paper_runs']==10
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
(LAB/'_delivery_l091_results.json').write_text('{"status":"RUNNING"}\n')
workflow=(ROOT/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[x[10:] for x in block.splitlines() if x.startswith('          ')]
lines=[x for x in lines if not x.startswith(('VER=','sed -i'))]
with tempfile.TemporaryDirectory(prefix='l091-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=ROOT,check=True,capture_output=True,text=True)
 checked=0
 for path in [stage/'lessons'/f'{SLUG}.html',stage/'reference/r-gcn.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   dest=(path.parent/unquote(part.path)).resolve();assert dest.exists(),(path,url)
   assert not dest.is_symlink();checked+=1
 for name in ['labs/solutions/'+SLUG+'.ipynb','labs/relkit/rgcn_l091.py','labs/_sources_l091.json','labs/_paper_l091_results.json','labs/_run_l091.py','labs/data/l091/aifb.tgz']:
  assert (stage/name).exists(),name
result={'status':'PASS','browser_widths':[1200,375],'typed_edge_intervention_and_keyboard':'PASS','portable_figures':4,'student_TODOs':3,'print_render':'PASS','actual_pages_copy_commands':'PASS','copied_local_links':checked,'executed_solution_sha256':hashlib.sha256((LAB/'solutions'/f'{SLUG}.ipynb').read_bytes()).hexdigest(),'page_errors':errors,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(LAB/'_delivery_l091_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
