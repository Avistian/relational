"""Real browser, source parity, portable notebook, local links and manifest delivery."""
import ast,base64,functools,hashlib,json,os,threading
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0185-causal-relational-data'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
studentcode='\n\n'.join(c.source for c in student.cells if c.cell_type=='code')
assert studentcode.count('raise NotImplementedError')==3 and 'raise NotImplementedError' not in code
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None for c in solution.cells if c.cell_type=='code')
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_l185_results.json').read_text())['code_sha256']
defs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
for n in ast.parse((P/'relkit/causal_l185.py').read_text()).body:
 if isinstance(n,ast.FunctionDef):assert defs[n.name]==ast.dump(n,include_attributes=False),n.name
for book in (student,solution):
 md='\n'.join(c.source for c in book.cells if c.cell_type=='markdown')
 assert md.count('data:image/png;base64,')==3 and 'attachment:' not in md
 import re
 for payload in re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',md):assert base64.b64decode(payload).startswith(b'\x89PNG')
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ('href','src'):self.links.append(v)
count=0
for f in [R/'lessons'/(S+'.html'),R/'reference/causal-relational-data.html',P/'html'/(S+'.html')]:
 parser=Links();parser.feed(f.read_text())
 for link in parser.links:
  u=urlsplit(link)
  if u.scheme or not u.path:continue
  path=(f.parent/unquote(u.path)).resolve()
  assert path.exists(),(f,link);count+=1
errors=[];states=0
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*a):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)))
threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}'
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('requestfailed',lambda q:errors.append(q.url))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':1000});page.goto(base+'/lessons/'+S+'.html')
  host=page.locator('#intervention')
  for target,value,expected in [('B','0',.5),('B','1',.5),('A','0',.475),('A','1',.525)]:
   host.locator('[data-target]').select_option(target);host.locator('[data-value]').select_option(value)
   assert abs(float(host.locator('output').get_attribute('data-risk'))-expected)<1e-12
   assert '50.0%' in host.locator('.baseline').inner_text();states+=1
  host.locator('[data-reset]').click();assert host.locator('[data-target]').input_value()=='B'
  host.locator('[data-target]').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert host.locator('[data-target]').input_value()=='A'
  host.screenshot(path=f'/tmp/l185-explorer-{width}.png')
  pred=page.locator('#predict');assert pred.locator('.predict-reveal').is_disabled()
  assert len(set(len(x.split()) for x in pred.locator('.predict-option').all_inner_texts()))==1
  pred.locator('[data-value=same]').click();pred.locator('.predict-reveal').click()
  assert 'unchanged' in pred.locator('.predict-outcome').inner_text()
  assert page.locator('#warmup button').count()==0 and page.locator('#teachback textarea').count()==1
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===3&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),f'Overflow {width}'
  for i in range(3):page.locator('figure').nth(i).screenshot(path=f'/tmp/l185-figure-{i}-{width}.png')
  page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l185-top-{width}.png')
 page.emulate_media(media='print');assert host.locator('.controls').evaluate('(x)=>getComputedStyle(x).display')=='none'
 page.screenshot(path='/tmp/l185-print.png');page.emulate_media(media='screen')
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto(base+'/lessons/'+S+'.html')
 assert 'Static intervention reference' in pg.locator('body').inner_text()
 assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1')
 pg.screenshot(path='/tmp/l185-nojs.png');context.close()
 page.set_viewport_size({'width':1200,'height':900});page.goto(base+'/labs/html/'+S+'.html')
 assert page.locator('img[src^="data:image/png"]').count()==3
 assert 'CHECK: all five seeds and causal bounds passed' in page.locator('body').inner_text()
 page.locator('img[src^="data:image/png"]').first.screenshot(path='/tmp/l185-notebook-figure.png')
 for url,selector in [('/index.html',f'a[href="lessons/{S}.html"]'),('/notebooks.html',f'a[href="labs/{S}.ipynb"]')]:
  page.goto(base+url);reveal_gallery_link(page,selector)
 assert not errors,errors
 browser.close()
server.shutdown()
result={'status':'PASS','browser_states':states,'viewports':[1200,375],'keyboard_reset':'PASS','print_nojs':'PASS','local_links_checked':count,'inline_source_parity':'PASS','portable_pngs_per_notebook':3,'student_live_tasks':3,'manifest_navigation':'PASS','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/'_delivery_l185_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
