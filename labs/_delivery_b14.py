"""Blank learner rejection and actual browser interaction/print/link validation."""
import functools,http.server,json,os,re,subprocess,tempfile,threading,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
P=Path(__file__).resolve().parent;R=P.parent;V=R/'reviews/lesson-b14';S='b14-flattening-challenge';start=time.monotonic();V.mkdir(parents=True,exist_ok=True)
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
n=nbformat.read(P/f'{S}.ipynb',4);assert sum('raise NotImplementedError("Complete' in c.source for c in n.cells if c.cell_type=='code')==3;assert all(not c.get('outputs') for c in n.cells if c.cell_type=='code')
assert all(not re.search(r'\{\{[A-Z_]+\}\}',c.source) for c in n.cells if c.cell_type=='markdown')
with tempfile.TemporaryDirectory(prefix='b14-blank-') as td:
 try:NotebookClient(n,timeout=90,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
 except CellExecutionError as e:assert 'Complete flatten' in str(e)
 else:raise AssertionError('Blank learner passed')
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}/';errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('response',lambda r:errors.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto(base+f'lessons/{S}.html')
  board=page.locator('[data-flat-time]')
  for snap in [8,10,14]:
   for arrival in ['check','ignore']:
    board.locator('[name=snapshot]').select_option(str(snap));board.locator('[name=arrival]').select_option(arrival)
    ev=[(2,3,2),(6,7,6),(7,12,100),(10,10,20)];vals=[v for t,a,v in ev if t<snap and (arrival=='ignore' or a<=snap)]
    assert abs(float(board.locator('output').get_attribute('data-mean'))-sum(vals)/len(vals))<1e-12
    assert int(board.locator('output').get_attribute('data-count'))==len(vals);states+=1
  board.locator('button').click();assert board.locator('output').get_attribute('data-mean')=='4'
  board.locator('[name=snapshot]').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert board.locator('[name=snapshot]').input_value()=='14';board.locator('button').click()
  swap=page.locator('[data-flat-swap]')
  for mode in ['backbone','features','context','snapshot']:
   swap.locator('select').select_option(mode);assert swap.locator('output').get_attribute('data-controlled')==str(mode=='backbone').lower();states+=1
  swap.locator('button').click();assert swap.locator('select').input_value()=='backbone'
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2'),'Horizontal overflow'
  assert page.locator('#b14-warmup').inner_text().strip();assert page.locator('#b14-predict').inner_text().strip();assert page.locator('#b14-teachback textarea').count()==1
  assert page.locator('img').evaluate_all('(xs)=>xs.every(x=>x.complete&&x.naturalWidth>0)')
  page.screenshot(path=str(V/f'lesson-{width}.png'),full_page=True)
  for i in range(3):page.locator('.flat-mobile' if width==375 else '.flat-wide').nth(i).screenshot(path=str(V/f'architecture-{i}-{width}.png'))
  board.screenshot(path=str(V/f'time-{width}.png'));swap.screenshot(path=str(V/f'swap-{width}.png'))
  if width==375:assert page.locator('[data-mobile-results] li').count()==12;page.locator('[data-mobile-results]').screenshot(path=str(V/'results-375.png'))
 page.emulate_media(media='print');assert page.locator('.flat-wide').first.is_visible();assert not board.locator('button').is_visible();page.screenshot(path=str(V/'print.png'),full_page=True);page.emulate_media(media='screen')
 nojs=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});q=nojs.new_page();q.goto(base+f'lessons/{S}.html');assert 'Controls need JavaScript' in q.locator('body').inner_text();assert 'mean 32' in q.locator('body').inner_text();nojs.close()
 page.goto(base+'index.html');page.wait_for_timeout(300);assert page.locator('a[href="lessons/'+S+'.html"]').count()>=1
 page.goto(base+'notebooks.html');page.wait_for_timeout(300);assert S in page.content()
 page.set_viewport_size({'width':1000,'height':900});page.goto(base+f'labs/html/{S}.html');assert page.locator('img').count()>=4;page.locator('img').nth(1).screenshot(path=str(V/'notebook-architecture.png'))
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
out=dict(status='PASS',blank_student='REJECTED',browser_states=states,keyboard_reset='PASS',desktop_mobile='PASS',print_nojs='PASS',local_links=count,browser_errors=errors,seconds=time.monotonic()-start,live_colab='NOT_CHECKED');(P/'_delivery_b14_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
