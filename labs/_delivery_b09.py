"""Browser controls, mobile geometry, links and blank-student checks."""
import functools,hashlib,http.server,json,os,tempfile,threading,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
P=Path(__file__).resolve().parent;R=P.parent;V=R/'reviews/lesson-b09';V.mkdir(exist_ok=True);S='b09-cost-frontier';start=time.monotonic()
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
assert sum('NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert all('NotImplementedError' not in c.source for c in solution.cells if c.cell_type=='code')
with tempfile.TemporaryDirectory(prefix='b09-blank-') as td:
 try:NotebookClient(student,timeout=60,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
 except CellExecutionError as e:assert 'Complete aligned_rmse' in str(e)
 else:raise AssertionError('Blank student passed')
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*a):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}/';errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('response',lambda r:errors.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto(base+f'lessons/{S}.html')
  cost=page.locator('[data-b09=cost]');att=page.locator('[data-b09=attention]')
  for n in [1,10,100]:
   for cache in [False,True]:
    cost.locator('input[type=range]').fill(str(n));cost.locator('input[type=checkbox]').set_checked(cache);cost.locator('input[type=range]').dispatch_event('input')
    assert abs(float(cost.locator('output').get_attribute('data-a'))-(6+n*.02))<1e-9
    assert abs(float(cost.locator('output').get_attribute('data-b'))-((1 if cache else n)+n*.08))<1e-9;states+=1
  cost.locator('button').click();assert cost.locator('input[type=range]').input_value()=='1'
  cost.locator('input[type=range]').focus();page.keyboard.press('ArrowRight');assert cost.locator('input[type=range]').input_value()=='2';cost.locator('button').click()
  for q in [10,20,30]:
   for illegal in [False,True]:
    att.locator('input[type=range]').fill(str(q));att.locator('input[type=checkbox]').set_checked(illegal);att.locator('input[type=range]').dispatch_event('input')
    assert float(att.locator('output').get_attribute('data-value'))==((18+q)/4 if illegal else 4);states+=1
  att.locator('button').click()
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2'),'Horizontal document overflow'
  page.screenshot(path=str(V/f'lesson-{width}.png'),full_page=True)
  cost.screenshot(path=str(V/f'cost-{width}.png'))
 page.emulate_media(media='print');page.screenshot(path=str(V/'print.png'),full_page=True)
 page.emulate_media(media='screen');page.goto(base+'index.html');page.locator('#lesson-nav a[href="lessons/'+S+'.html"]').wait_for()
 page.goto(base+'notebooks.html');page.locator('#lab-B09').wait_for();assert page.locator('#nb-list li').first.get_attribute('id')=='lab-B09'
 page.goto(base+f'labs/html/{S}.html');assert page.locator('img').count()>=1;page.screenshot(path=str(V/'notebook.png'))
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});p=context.new_page();p.goto(base+f'lessons/{S}.html');assert 'Baseline, one request' in p.locator('body').inner_text();context.close();browser.close()
server.shutdown();assert not errors,errors
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,t,a):self.links.extend(v for k,v in a if k in ['href','src'])
count=0
for file in [R/f'lessons/{S}.html',R/f'reference/{S}.html']:
 p=Links();p.feed(file.read_text())
 for url in p.links:
  u=urlsplit(url)
  if u.scheme or not u.path:continue
  assert (file.parent/unquote(u.path)).resolve().is_file(),url;count+=1
r=dict(status='PASS',browser_states=states,desktop_and_375px=True,keyboard_reset=True,print_and_nojs=True,blank_student_refused=True,local_links=count,seconds=time.monotonic()-start,live_colab='NOT_CHECKED',deployment='NOT_RUN')
(P/'_delivery_b09_results.json').write_text(json.dumps(r,indent=2));print(r)
