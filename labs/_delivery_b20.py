"""Learner failures, paper gate/corruption, actual browser and portable figure checks."""
import copy,functools,http.server,json,os,re,shutil,subprocess,sys,tempfile,threading,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
P=Path(__file__).resolve().parent;R=P.parent;V=R/'reviews/lesson-b20';S='b20-curriculum-order';started=time.monotonic()
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
indices=[i for i,c in enumerate(student.cells) if c.cell_type=='code' and c.metadata.get('learner_function')];assert len(indices)==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert len(re.findall(r'data:image/png;base64,','\n'.join(c.source for c in student.cells)))==4
for index in indices:
 n=copy.deepcopy(student)
 for j in indices:
  if j!=index:n.cells[j].source=solution.cells[j].source
 with tempfile.TemporaryDirectory(prefix='b20-blank-') as td:
  try:NotebookClient(n,timeout=90,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
  except CellExecutionError as e:assert 'NotImplementedError' in str(e)
  else:raise AssertionError('Blank task passed')
import _test_b20 as tests
for name,test,bad in [('exposure_order','test_order',lambda levels,mode,seed,repeats=2:list(range(len(levels)))),('paired_contract','test_pair',lambda a,b:True),('auc_retention','test_retention',lambda s,b:dict(raw=s/b,above_chance=s/b))]:
 old=getattr(tests,name);setattr(tests,name,bad)
 try:tests.CurriculumTests(test).debug()
 except (AssertionError,ValueError):pass
 else:raise AssertionError('Wrong function passed '+name)
 finally:setattr(tests,name,old)
with tempfile.TemporaryDirectory(prefix='b20-corrupt-') as td:
 t=Path(td);shutil.copytree(P/'sources/b20',t/'sources/b20');shutil.copy(P/'_reproduce_b20.py',t/'_reproduce_b20.py')
 proc=subprocess.run([sys.executable,str(t/'_reproduce_b20.py'),'--fresh'],capture_output=True,text=True);assert proc.returncode!=0 and 'NOT_RUN:' in proc.stderr
 f=t/'sources/b20/paper.html';f.write_bytes(f.read_bytes()+b'corrupt')
 proc=subprocess.run([sys.executable,str(t/'_reproduce_b20.py')],capture_output=True,text=True);assert proc.returncode!=0 and 'SOURCE_HASH_MISMATCH' in proc.stderr
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}/';errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('requestfailed',lambda req:errors.append(req.url));page.on('response',lambda response:errors.append(response.url) if response.url.startswith(base) and response.status>=400 else None)
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto(base+f'lessons/{S}.html');board=page.locator('[data-curriculum-order]')
  for step in range(5):
   for rate in ['.025','.1','.2']:
    board.locator('[name=step]').select_option(str(step));board.locator('[name=rate]').select_option(rate)
    output=board.locator('output').inner_text();assert board.locator('.used').count()==2*step
    for order in [[(1,1),(1,1),(2,-1),(2,-1)],[(1,1),(2,-1),(1,1),(2,-1)]]:
     w=0
     for x,y in order[:step]:w-=float(rate)*x*(w*x-y)
     loss=sum(.5*(w*x-y)**2 for x,y in order)/4
     assert f'{w:.6f}' in output and f'{loss:.6f}' in output
    states+=1
  board.locator('button').click();assert board.locator('[name=step]').input_value()=='4';assert board.locator('[name=rate]').input_value()=='.1'
  board.locator('[name=step]').focus();page.keyboard.press('ArrowUp');page.keyboard.press('Enter');assert board.locator('[name=step]').input_value()=='3';board.locator('button').click()
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2'),'Page horizontal overflow'
  assert page.locator('#b20-warmup').inner_text().strip()
  page.screenshot(path=str(V/f'lesson-{width}.png'),full_page=True);board.screenshot(path=str(V/f'order-{width}.png'))
  for i,fig in enumerate(page.locator('.curriculum-figure').all()):
   if width==375:assert fig.evaluate('(el)=>el.scrollWidth>el.clientWidth');fig.focus();page.keyboard.press('ArrowRight');fig.evaluate('(el)=>el.scrollLeft=0')
   fig.screenshot(path=str(V/f'figure-{i}-{width}.png'))
  for choice in ['no','yes']:
   page.goto(base+f'lessons/{S}.html');pred=page.locator('#b20-predict');assert pred.locator('.predict-reveal').is_disabled();pred.locator('[data-value='+choice+']').click();pred.locator('.predict-reveal').click();assert pred.locator('.predict-outcome').inner_text().strip();states+=1
  teach=page.locator('#b20-teachback');teach.locator('textarea').fill('Freeze task identities and repetitions, initialization and persistent optimizer, budget and final selection. Different database counts do not imply different compute. Arithmetic replay cannot identify the historical trained model.');teach.locator('button').first.click();assert 'I freeze' in teach.inner_text()
  page.goto(base+f'labs/html/{S}.html');assert page.locator('img[src^="data:image/png"]').count()>=4;assert page.evaluate('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)')
  for i,fig in enumerate(page.locator('.curriculum-figure').all()):fig.screenshot(path=str(V/f'notebook-figure-{i}-{width}.png'))
 page.goto(base+'index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]');page.goto(base+'notebooks.html');page.wait_for_selector('a[href="labs/'+S+'.ipynb"]')
 page.goto(base+f'lessons/{S}.html');page.emulate_media(media='print');assert page.locator('.curriculum-controls').is_hidden()
 assert page.locator('.curriculum-figure img').first.evaluate('(el)=>getComputedStyle(el).minWidth')=='0px'
 page.screenshot(path=str(V/'print.png'),full_page=True)
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});p=context.new_page();p.goto(base+f'lessons/{S}.html');assert '−0.2516' in p.locator('output').inner_text();assert p.locator('noscript').inner_text();p.screenshot(path=str(V/'nojs.png'),full_page=True);browser.close()
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
out=dict(status='PASS',blank_tasks_rejected=3,wrong_tasks_rejected=3,source_corruption_rejected=True,fresh_paper_gate=True,browser_states=states,desktop_and_375px=True,keyboard_reset=True,no_js=True,print=True,portable_figures=4,local_links=links,browser_errors=errors,seconds=time.monotonic()-started)
(P/'_delivery_b20_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
