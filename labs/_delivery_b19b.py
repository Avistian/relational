"""Reject blank/wrong learner functions and inspect real desktop/mobile interaction."""
import base64,copy,functools,http.server,json,os,re,shutil,subprocess,sys,tempfile,threading,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
P=Path(__file__).resolve().parent;R=P.parent;V=R/'reviews/lesson-b19b';S='b19b-forecasting-contracts';start=time.monotonic()
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
indices=[i for i,c in enumerate(student.cells) if c.cell_type=='code' and 'raise NotImplementedError("Implement' in c.source];assert len(indices)==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
images=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)','\n'.join(c.source for c in student.cells));assert len(images)==4
for index in indices:
 n=copy.deepcopy(student)
 for j in indices:
  if j!=index:n.cells[j].source=solution.cells[j].source
 with tempfile.TemporaryDirectory(prefix='b19b-blank-') as td:
  try:NotebookClient(n,timeout=90,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
  except CellExecutionError as e:assert 'NotImplementedError' in str(e)
  else:raise AssertionError('Blank learner function passed')
import _test_b19b as tests
for name,test,bad in [('available_features','test_availability',lambda rows,o:{r['id']:r['value'] for r in rows if r['time']<=o}),('rolling_origins','test_origins',lambda n,o,h,s:[(i,list(range(i,min(i+h,n)))) for i in range(o,n,s)]),('forecast_scores','test_scores',lambda *args:dict(mase=1.,wql=1.,scale=1.))]:
 old=getattr(tests,name);setattr(tests,name,bad)
 try:tests.ForecastTests(test).debug()
 except (AssertionError,KeyError):pass
 else:raise AssertionError('Wrong learner function passed')
 finally:setattr(tests,name,old)
with tempfile.TemporaryDirectory(prefix='b19b-source-') as td:
 t=Path(td);shutil.copytree(P/'sources/b19b',t/'sources/b19b');shutil.copy(P/'_reproduce_b19b.py',t/'_reproduce_b19b.py')
 blocked=subprocess.run([sys.executable,str(t/'_reproduce_b19b.py'),'--fresh'],capture_output=True,text=True);assert blocked.returncode!=0 and 'NOT_RUN:' in blocked.stderr
 f=t/'sources/b19b/paper.html';f.write_bytes(f.read_bytes()+b'corruption')
 bad=subprocess.run([sys.executable,str(t/'_reproduce_b19b.py')],capture_output=True,text=True);assert bad.returncode!=0 and 'SOURCE_HASH_MISMATCH' in bad.stderr
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}/';errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('requestfailed',lambda req:errors.append(req.url));page.on('response',lambda response:errors.append(response.url) if response.url.startswith(base) and response.status>=400 else None)
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto(base+f'lessons/{S}.html');board=page.locator('[data-forecast-availability]')
  for issue in [10,11,12]:
   for weather in [8,12]:
    board.locator('[name=issue]').select_option(str(issue));board.locator('[name=weather]').select_option(str(weather));expected=2+int(issue>=11)+int(issue>=weather)
    assert int(board.locator('output').get_attribute('data-count'))==expected;assert board.locator('.forecast-cell.known').count()==expected;states+=1
  board.locator('button').click();assert board.locator('[name=issue]').input_value()=='10';assert board.locator('[name=weather]').input_value()=='12'
  board.locator('[name=issue]').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert board.locator('[name=issue]').input_value()=='11';board.locator('button').click()
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2'),'Horizontal overflow'
  assert page.locator('#b19b-warmup').inner_text().strip();assert page.locator('#b19b-predict').inner_text().strip();assert page.locator('#b19b-teachback textarea').count()==1
  page.screenshot(path=str(V/f'lesson-{width}.png'),full_page=True);board.screenshot(path=str(V/f'availability-{width}.png'))
  for i,fig in enumerate(page.locator('.forecast-figure').all()):
   if width==375:
    assert fig.evaluate('(el)=>el.scrollWidth>el.clientWidth');fig.focus();page.keyboard.press('ArrowRight');fig.evaluate('(el)=>el.scrollLeft=0')
   fig.screenshot(path=str(V/f'figure-{i}-{width}.png'))
  for value in ['oracle','deploy']:
   page.goto(base+f'lessons/{S}.html');pred=page.locator('#b19b-predict');assert pred.locator('.predict-reveal').is_disabled();pred.locator('[data-value='+value+']').click();pred.locator('.predict-reveal').click();assert pred.locator('.predict-outcome').inner_text().strip();states+=1
  teach=page.locator('#b19b-teachback');teach.locator('textarea').fill('Availability must precede the issue time; future labels cannot fit seasonal features. A saved-score replay does not authenticate historical inference.');teach.locator('button').first.click();assert 'Tomorrow' in teach.inner_text()
  # All embedded notebook figures actually render, not just valid PNG headers.
  page.goto(base+f'labs/html/{S}.html');assert page.locator('img[src^="data:image/png"]').count()>=4
  assert page.evaluate('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)')
  for i,fig in enumerate(page.locator('.forecast-figure').all()):
   if width==375:assert fig.evaluate('(el)=>el.scrollWidth>el.clientWidth')
   fig.screenshot(path=str(V/f'notebook-figure-{i}-{width}.png'))
  page.screenshot(path=str(V/f'notebook-{width}.png'),full_page=True)
 page.goto(base+'index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]');page.goto(base+'notebooks.html');page.wait_for_selector('a[href="labs/'+S+'.ipynb"]')
 page.goto(base+f'lessons/{S}.html');page.emulate_media(media='print');assert page.locator('.evidence-controls').is_hidden();page.screenshot(path=str(V/'print.png'),full_page=True)
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});p=context.new_page();p.goto(base+f'lessons/{S}.html');assert 'calendar and promotion available' in p.locator('output').inner_text();assert p.locator('noscript').inner_text();p.screenshot(path=str(V/'nojs.png'),full_page=True);browser.close()
server.shutdown();assert not errors,errors
class Links(HTMLParser):
 def __init__(self):super().__init__();self.urls=[]
 def handle_starttag(self,tag,attrs):self.urls.extend(v for k,v in attrs if k in ('href','src'))
links=0
for file in [R/'lessons'/f'{S}.html',R/'reference'/f'{S}.html']:
 parser=Links();parser.feed(file.read_text())
 for url in parser.urls:
  u=urlsplit(url)
  if u.scheme:continue
  target=(file.parent/unquote(u.path)).resolve() if u.path else file
  assert target.is_file(),url;links+=1
out=dict(status='PASS',blank_tasks_rejected=3,wrong_tasks_rejected=3,source_corruption_rejected=True,unauthorized_fresh_mode_rejected=True,browser_states=states,desktop_and_375px=True,keyboard_reset=True,no_js=True,print=True,portable_figures=4,local_links=links,browser_errors=errors,seconds=time.monotonic()-start)
(P/'_delivery_b19b_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
