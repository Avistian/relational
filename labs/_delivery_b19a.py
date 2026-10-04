"""Learner failure, source authentication and rendered delivery checks."""
import base64,copy,functools,http.server,json,os,re,shutil,subprocess,sys,tempfile,threading,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
P=Path(__file__).resolve().parent;R=P.parent;V=R/'reviews/lesson-b19a';S='b19a-predictive-distributions';start=time.monotonic()
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
r=json.loads((P/'evidence/b19a/diagnostic.json').read_text());student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
indices=[i for i,c in enumerate(student.cells) if c.cell_type=='code' and 'raise NotImplementedError("Implement' in c.source];assert len(indices)==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
images=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)','\n'.join(c.source for c in student.cells));assert len(images)==3
assert all(base64.b64decode(x).startswith(b'\x89PNG\r\n\x1a\n') for x in images)
for index in indices:
 n=copy.deepcopy(student)
 for j in indices:
  if j!=index:n.cells[j].source=solution.cells[j].source
 with tempfile.TemporaryDirectory(prefix='b19a-blank-') as td:
  try:NotebookClient(n,timeout=90,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
  except CellExecutionError as e:assert 'NotImplementedError' in str(e)
  else:raise AssertionError('Blank learner function passed')
import _test_b19a as tests
for name,test,bad in [('crps','test_crps',lambda v,p,y:sum(w*abs(x-y) for x,w in zip(v,p))),('interval_score','test_interval',lambda l,u,y,a:u-l),('calibrate','test_calibration',lambda rows,a:dict(n=len(rows),k=len(rows),radius=max(abs(x['y']-x['mean']) for x in rows),ids=[x['id'] for x in rows]))]:
 old=getattr(tests,name);setattr(tests,name,bad)
 try:tests.ScoringTests(test).debug()
 except (AssertionError,KeyError):pass
 else:raise AssertionError('Wrong learner function passed')
 finally:setattr(tests,name,old)
with tempfile.TemporaryDirectory(prefix='b19a-source-') as td:
 t=Path(td);shutil.copytree(P/'sources/b19a',t/'sources/b19a');shutil.copy(P/'_reproduce_b19a.py',t/'_reproduce_b19a.py')
 blocked=subprocess.run([sys.executable,str(t/'_reproduce_b19a.py'),'--fresh'],capture_output=True,text=True);assert blocked.returncode!=0 and 'NOT_RUN:' in blocked.stderr
 (P/'evidence/b19a/fresh-refusal.txt').write_text(blocked.stdout+blocked.stderr)
 f=t/'sources/b19a/paper.html';f.write_bytes(f.read_bytes()+b'corruption')
 bad=subprocess.run([sys.executable,str(t/'_reproduce_b19a.py')],capture_output=True,text=True);assert bad.returncode!=0 and 'SOURCE_HASH_MISMATCH' in bad.stderr
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}/';errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('requestfailed',lambda req:errors.append(req.url));page.on('response',lambda response:errors.append(response.url) if response.url.startswith(base) and response.status>=400 else None)
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto(base+f'lessons/{S}.html');board=page.locator('[data-distribution-board]')
  for name in r['forecasts']:
   for y in [-2,0,2]:
    board.locator('[name=forecast]').select_option(name);board.locator('[name=outcome]').select_option(str(y));expected=next(x for x in r['rows'] if x['forecast']==name and x['y']==y)
    assert float(board.locator('output').get_attribute('data-crps'))==expected['crps'];assert float(board.locator('output').get_attribute('data-interval'))==expected['interval_score'];assert board.locator('svg path').count()==3;states+=1
  board.locator('button').click();assert board.locator('output').get_attribute('data-key')=='narrow/0'
  board.locator('[name=forecast]').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert board.locator('[name=forecast]').input_value()=='calibrated';board.locator('button').click()
  cal=page.locator('[data-calibration-board]')
  for shift,expected in [('0',.5),('100',0)]:
   cal.locator('select').select_option(shift);assert cal.locator('output').get_attribute('data-radius')=='7';assert float(cal.locator('output').get_attribute('data-coverage'))==expected;states+=1
  cal.locator('button').click();assert cal.locator('select').input_value()=='0'
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2'),'Horizontal overflow'
  assert page.locator('#b19a-warmup').inner_text().strip();assert page.locator('#b19a-predict').inner_text().strip();assert page.locator('#b19a-teachback textarea').count()==1
  page.screenshot(path=str(V/f'lesson-{width}.png'),full_page=True)
  for selector,name in [('[data-distribution-board]','distribution'),('[data-calibration-board]','calibration'),('[aria-label="Distribution evaluation architecture"]','pipeline')]:page.locator(selector).screenshot(path=str(V/f'{name}-{width}.png'))
  for i,fig in enumerate(page.locator('.evidence-figure').all()):
   if width==375:
    assert fig.evaluate('(el)=>el.scrollWidth>el.clientWidth');fig.focus();page.keyboard.press('End');fig.evaluate('(el)=>el.scrollLeft=0')
   fig.screenshot(path=str(V/f'figure-{i}-{width}.png'))
 page.emulate_media(media='print');assert not board.locator('button').is_visible();assert board.locator('output').is_visible();page.screenshot(path=str(V/'print.png'),full_page=True);page.emulate_media(media='screen')
 nojs=browser.new_page(java_script_enabled=False);nojs.goto(base+f'lessons/{S}.html');assert nojs.locator('noscript').first.is_visible();assert nojs.locator('table').count()==2;nojs.close()
 page.goto(base+'index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]')
 page.goto(base+'notebooks.html');page.wait_for_selector('a[href="labs/'+S+'.ipynb"]')
 page.set_viewport_size({'width':1000,'height':950});page.goto(base+f'labs/html/{S}.html');assert page.locator('img[src^="data:image/png"]').count()==3;page.locator('img[src^="data:image/png"]').first.screenshot(path=str(V/'notebook-pipeline.png'))
 browser.close()
server.shutdown();assert not errors,errors
class Links(HTMLParser):
 def __init__(self):super().__init__();self.urls=[]
 def handle_starttag(self,tag,attrs):self.urls.extend(v for k,v in attrs if k in ('href','src'))
links=0
for path in [R/'lessons'/f'{S}.html',R/'reference'/f'{S}.html']:
 parser=Links();parser.feed(path.read_text())
 for url in parser.urls:
  u=urlsplit(url)
  if u.scheme:continue
  target=(path.parent/unquote(u.path)).resolve() if u.path else path
  assert target.is_file(),url;links+=1
out=dict(status='PASS',blank_tasks_rejected=3,wrong_functions_rejected=3,corrupted_source_rejected=True,fresh_refusal=True,desktop_mobile_states=states,keyboard_reset=True,nojs_print=True,portable_figures=3,local_links=links,browser_errors=errors,seconds=time.monotonic()-start,live_colab='NOT_CHECKED')
(P/'_delivery_b19a_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
