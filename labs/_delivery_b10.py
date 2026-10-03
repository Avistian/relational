"""Exercise every mask/reader and cutoff, inspect portable delivery and links."""
import functools,http.server,json,os,tempfile,threading,time,subprocess
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
from _test_b10 import fixture,oracle
P=Path(__file__).resolve().parent;R=P.parent;V=R/'reviews/lesson-b10';S='b10-relational-transformer';start=time.monotonic();V.mkdir(exist_ok=True)
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
assert sum('NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert all('NotImplementedError' not in c.source for c in solution.cells if c.cell_type=='code')
assert sum('data:image/png;base64,' in c.source for c in student.cells)==3
with tempfile.TemporaryDirectory(prefix='b10-blank-') as td:
 try:NotebookClient(student,timeout=90,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
 except CellExecutionError as e:assert 'Complete relational_masks' in str(e)
 else:raise AssertionError('Blank student passed')
# Running benchmark is expected to refuse, not silently fall back to a toy model.
gate=subprocess.run([str(R/'.venv/bin/python'),str(P/'_reproduce_b10.py'),'--run','--checkpoints','/nonexistent','--out','/nonexistent-b10'],capture_output=True,text=True)
assert gate.returncode!=0 and 'BLOCKED_TEMPORAL_AUDIT' in gate.stderr,gate.stderr
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*a):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}/';errors=[];states=0
truth=oracle(fixture());coords=[0,10,42,1,6,7,1,2,20]
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('response',lambda r:errors.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto(base+f'lessons/{S}.html')
  masks=page.locator('[data-b10=masks]');timeline=page.locator('[data-b10=time]')
  for kind in ['col','feat','nbr','full']:
   masks.locator('[name=kind]').select_option(kind)
   for i in range(9):
    masks.locator('[name=reader]').select_option(str(i));allowed=truth[kind][0,i,:9].tolist();count=sum(allowed);mean=sum(v for v,a in zip(coords,allowed) if a)/count if count else 0
    assert int(masks.locator('output').get_attribute('data-count'))==count
    assert abs(float(masks.locator('output').get_attribute('data-mean'))-mean)<1e-10
    assert masks.locator('.allowed').count()==count;states+=1
  masks.locator('button').click();assert masks.locator('[name=kind]').input_value()=='feat'
  for cutoff in range(8,15):
   timeline.locator('input').fill(str(cutoff));timeline.locator('input').dispatch_event('input')
   expected=1+int(cutoff>=11)+int(cutoff>=12)
   assert int(timeline.locator('output').get_attribute('data-count'))==expected;states+=1
  timeline.locator('button').click();timeline.locator('input').focus();page.keyboard.press('ArrowRight');assert timeline.locator('input').input_value()=='11';timeline.locator('button').click()
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2'),'Horizontal document overflow'
  page.screenshot(path=str(V/f'lesson-{width}.png'),full_page=True)
  masks.screenshot(path=str(V/f'masks-{width}.png'));timeline.screenshot(path=str(V/f'timeline-{width}.png'))
 page.emulate_media(media='print');page.screenshot(path=str(V/'print.png'),full_page=True)
 page.emulate_media(media='screen');page.goto(base+'index.html');page.locator('#lesson-nav a[href="lessons/'+S+'.html"]').wait_for()
 page.goto(base+'notebooks.html');page.locator('#lab-B10').wait_for();assert page.locator('#nb-list li').first.get_attribute('id')=='lab-B10'
 page.goto(base+f'labs/html/{S}.html');assert page.locator('img').count()>=3;page.screenshot(path=str(V/'notebook.png'))
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});p=context.new_page();p.goto(base+f'lessons/{S}.html');assert 'Default feature attention' in p.locator('body').inner_text();context.close();browser.close()
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
r=dict(status='PASS',browser_states=states,desktop_and_375px=True,keyboard_reset=True,print_and_nojs=True,blank_student_refused=True,benchmark_refused_before_loading=True,local_links=count,seconds=time.monotonic()-start,live_colab='NOT_CHECKED',deployment='NOT_RUN')
(P/'_delivery_b10_results.json').write_text(json.dumps(r,indent=2));print(r)
