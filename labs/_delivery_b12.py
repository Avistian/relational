"""Check student failure, live controls, mobile/no-JS/print, images and navigation."""
import functools,http.server,json,os,subprocess,tempfile,threading,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
P=Path(__file__).resolve().parent;R=P.parent;V=R/'reviews/lesson-b12';S='b12-adaptation-mechanisms';start=time.monotonic();V.mkdir(parents=True,exist_ok=True)
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/f'{S}.ipynb',4)
assert sum('raise NotImplementedError("Complete' in c.source for c in student.cells if c.cell_type=='code')==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert all('{{' not in c.source for c in student.cells if c.cell_type=='markdown')
with tempfile.TemporaryDirectory(prefix='b12-blank-') as td:
 try:NotebookClient(student,timeout=90,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
 except CellExecutionError as e:assert 'Complete eligible_support' in str(e)
 else:raise AssertionError('Blank student passed')
gate=subprocess.run([str(R/'.venv/bin/python'),str(P/'_reproduce_b12.py'),'--run-full'],capture_output=True,text=True)
assert gate.returncode!=0 and 'BLOCKED: USD0' in gate.stderr
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*a):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}/';errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('response',lambda r:errors.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto(base+f'lessons/{S}.html');board=page.locator('[data-adaptation-paths]')
  for reach in ['low','high']:
   for mode in ['intact','shuffled','hidden']:
    board.locator('[name=reach]').select_option(reach);board.locator('[name=mode]').select_option(mode)
    batch=.5 if mode=='hidden' else (.75 if mode=='intact' else .25);relation=batch if reach=='high' else .5
    assert float(board.locator('output').get_attribute('data-relation'))==relation
    assert float(board.locator('output').get_attribute('data-batch'))==batch
    assert float(board.locator('output').get_attribute('data-dual'))==(relation+batch)/2;states+=1
  board.locator('button').click();assert board.locator('[name=reach]').input_value()=='low' and board.locator('[name=mode]').input_value()=='intact'
  board.locator('[name=reach]').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert board.locator('[name=reach]').input_value()=='high'
  board.locator('button').click();assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2'),'Horizontal overflow'
  assert page.locator('#b12-warmup').inner_text().strip()
  assert page.locator('#b12-predict').inner_text().strip()
  assert page.locator('#b12-teachback textarea').count()==1
  assert page.locator('img').evaluate_all('(images)=>images.every(x=>x.complete&&x.naturalWidth>0)')
  page.screenshot(path=str(V/f'lesson-{width}.png'),full_page=True);page.locator('.adapt-mobile' if width==375 else '.adapt-wide').first.screenshot(path=str(V/f'architecture-{width}.png'));board.screenshot(path=str(V/f'trace-{width}.png'))
  if width==375:assert page.locator('.adapt-mobile').first.is_visible() and not page.locator('.adapt-wide').first.is_visible()
 page.emulate_media(media='print');assert page.locator('.adapt-wide').first.is_visible();page.screenshot(path=str(V/'print.png'),full_page=True);page.emulate_media(media='screen')
 page.goto(base+'index.html');page.locator('#lesson-nav a[href="lessons/'+S+'.html"]').wait_for()
 page.goto(base+'notebooks.html');page.locator('#lab-B12').wait_for()
 page.set_viewport_size({'width':1100,'height':950});page.goto(base+f'labs/html/{S}.html');assert page.locator('img').count()==6;page.screenshot(path=str(V/'notebook.png'))
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});p=context.new_page();p.goto(base+f'lessons/{S}.html');assert 'Dual average = 0.625' in p.locator('body').inner_text();context.close();browser.close()
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
r=dict(status='PASS',browser_states=states,desktop_and_375px=True,keyboard_reset=True,print_and_nojs=True,blank_student_refused=True,benchmark_refused_before_loading=True,local_links=count,portable_images=6,seconds=time.monotonic()-start,live_colab='NOT_CHECKED',deployment='NOT_RUN')
(P/'_delivery_b12_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
