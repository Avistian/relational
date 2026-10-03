"""Actual browser state audit plus blank learner and local link checks."""
import functools,http.server,json,os,tempfile,threading,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
P=Path(__file__).resolve().parent;R=P.parent;V=R/'reviews/lesson-b15';S='b15-parameter-free-encoders';start=time.monotonic();V.mkdir(parents=True,exist_ok=True)
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
n=nbformat.read(P/f'{S}.ipynb',4);assert sum('raise NotImplementedError("Complete' in c.source for c in n.cells if c.cell_type=='code')==3
assert all(not c.get('outputs') for c in n.cells if c.cell_type=='code')
with tempfile.TemporaryDirectory(prefix='b15-blank-') as td:
 try:NotebookClient(n,timeout=90,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
 except CellExecutionError as e:assert 'Complete visible_labels' in str(e)
 else:raise AssertionError('Blank learner passed')
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}/';errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('response',lambda r:errors.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto(base+f'lessons/{S}.html')
  v=page.locator('[data-label-visibility]')
  for scope in ['local','expanded']:
   for cutoff in [3,10,11]:
    v.locator('[name=scope]').select_option(scope);v.locator('[name=cutoff]').select_option(str(cutoff))
    assert v.locator('output').get_attribute('data-remote')==str(scope=='expanded' and cutoff>=4).lower()
    assert v.locator('output').get_attribute('data-future')==str(scope=='expanded' and cutoff>=11).lower();states+=1
  v.locator('button').click();assert v.locator('[name=cutoff]').input_value()=='10'
  v.locator('[name=cutoff]').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert v.locator('[name=cutoff]').input_value()=='11';v.locator('button').click()
  r=page.locator('[data-label-rules]')
  for world in [0,1]:
   for access in ['local','expanded']:
    r.locator('[name=world]').select_option(str(world));r.locator('[name=access]').select_option(access)
    assert float(r.locator('output').get_attribute('data-probability'))==(.5 if access=='local' else 1-world);states+=1
  r.locator('button').click();assert r.locator('[name=access]').input_value()=='local'
  c=page.locator('[data-label-columns]')
  for col in [0,1]:
   for access in ['local','expanded']:
    c.locator('[name=column]').select_option(str(col));c.locator('[name=examples]').select_option(access)
    assert float(c.locator('output').get_attribute('data-probability'))==(.5 if access=='local' else col)
    assert int(c.locator('output').get_attribute('data-bits'))==int(access=='expanded');states+=1
  c.locator('button').click();assert c.locator('[name=examples]').input_value()=='local'
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2'),'Horizontal overflow'
  assert page.locator('#b15-warmup').inner_text().strip();assert page.locator('#b15-predict').inner_text().strip();assert page.locator('#b15-teachback textarea').count()==1
  page.screenshot(path=str(V/f'lesson-{width}.png'),full_page=True)
  for name,board in [('visibility',v),('rules',r),('columns',c)]:board.screenshot(path=str(V/f'{name}-{width}.png'))
 page.emulate_media(media='print');assert not v.locator('button').is_visible();page.screenshot(path=str(V/'print.png'),full_page=True);page.emulate_media(media='screen')
 nojs=browser.new_page(java_script_enabled=False);nojs.goto(base+f'lessons/{S}.html');assert nojs.locator('noscript').first.is_visible();nojs.close()
 page.goto(base+'index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]')
 page.goto(base+'notebooks.html');page.wait_for_timeout(300);assert S in page.content()
 page.set_viewport_size({'width':1000,'height':900});page.goto(base+f'labs/html/{S}.html');assert page.locator('img').count()>=1;page.locator('img').first.screenshot(path=str(V/'notebook-information.png'))
 browser.close()
server.shutdown();assert not errors,errors
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,t,a):self.links.extend(v for k,v in a if k in ('href','src'))
count=0
for name in ['lessons/'+S+'.html','reference/'+S+'.html']:
 path=R/name;p=Links();p.feed(path.read_text())
 for url in p.links:
  u=urlsplit(url)
  if u.scheme or not u.path:continue
  assert (path.parent/unquote(u.path)).resolve().is_file(),url;count+=1
out=dict(status='PASS',blank_student='REJECTED',browser_states=states,keyboard_reset='PASS',desktop_mobile='PASS',print_nojs='PASS',local_links=count,browser_errors=errors,seconds=time.monotonic()-start,live_colab='NOT_CHECKED');(P/'_delivery_b15_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
