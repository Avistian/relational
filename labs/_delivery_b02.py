"""Actual browser, portable-source, deterministic build and link checks."""
import ast,functools,hashlib,http.server,json,os,subprocess,sys,tempfile,threading,zipfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';E=P/'evidence/b02';S='b02-numerical-embeddings-and-ensembles';V=R/'reviews/lesson-b02';V.mkdir(exist_ok=True)
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution_path=P/'solutions'/(S+'.ipynb');solution=nbformat.read(solution_path,4)
sc='\n\n'.join(c.source for c in student.cells if c.cell_type=='code');code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert sc.count('NotImplementedError')==3 and 'NotImplementedError' not in code
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert json.loads((P/'_execution_b02_results.json').read_text())['executed_code_sha256']==hashlib.sha256(code.encode()).hexdigest()
functions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
for name in ['relkit/embeddings_b02.py','_audit_b02.py','_test_b02.py']:
 for n in ast.parse((P/name).read_text()).body:
  if isinstance(n,ast.FunctionDef) and n.name not in ['reserve_budget','check_budget']:assert functions[n.name]==ast.dump(n,include_attributes=False),n.name
paths=[R/'lessons'/(S+'.html'),R/'reference/b02-embeddings-ensembles.html',P/(S+'.ipynb'),E/'replay.zip']
before={p:p.read_bytes() for p in paths};executed=solution_path.read_bytes()
try:
 subprocess.run([sys.executable,str(P/'_build_b02.py')],check=True,capture_output=True)
 rebuilt=nbformat.read(solution_path,4);assert '\n\n'.join(c.source for c in rebuilt.cells if c.cell_type=='code')==code
 assert all(p.read_bytes()==b for p,b in before.items()),'Non-deterministic builder'
finally:solution_path.write_bytes(executed)
with tempfile.TemporaryDirectory(prefix='b02-replay-') as td:
 with zipfile.ZipFile(E/'replay.zip') as z:z.extractall(td)
 sys.path.insert(0,td)
 from _audit_b02 import audit
 assert audit(Path(td)/'compact')==json.loads((E/'report.json').read_text())
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*a):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)))
threading.Thread(target=server.serve_forever,daemon=True).start();base='http://127.0.0.1:'+str(server.server_port)+'/'
errors=[];states=0
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.on('response',lambda r:errors.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
  for width in [1200,375]:
   page.set_viewport_size({'width':width,'height':950});page.goto(base+'lessons/'+S+'.html')
   assert all(marker not in page.locator('body').inner_text() for marker in ['[[RESULTS]]','[[WIDGET]]','[[ARCHITECTURE]]'])
   for val,ids,valid in [('validation','0,1','true'),('test','2','false')]:
    page.get_by_label('Labels used for selection').select_option(val)
    assert page.locator('output').get_attribute('data-ids')==ids
    assert page.locator('output').get_attribute('data-valid')==valid;states+=1
   page.get_by_role('button',name='Reset').click();assert page.locator('output').get_attribute('data-valid')=='true'
   page.get_by_label('Labels used for selection').focus();page.keyboard.press('ArrowDown');assert page.locator('output').get_attribute('data-valid')=='false'
   page.get_by_role('button',name='Reset').click()
   assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),width
   assert page.locator('.b02-architecture img').evaluate('(el)=>el.complete&&el.naturalWidth>0')
   page.evaluate('scrollTo(0,0)');page.screenshot(path=str(V/f'top-{width}.png'))
   page.locator('.b02-widget').screenshot(path=str(V/f'widget-{width}.png'))
   page.locator('.b02-architecture').screenshot(path=str(V/f'architecture-{width}.png'))
  page.emulate_media(media='print');assert page.locator('.b02-widget select').evaluate('(el)=>getComputedStyle(el).display')=='none';page.emulate_media(media='screen')
  context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});pg=context.new_page();pg.goto(base+'lessons/'+S+'.html')
  assert 'VALID SELECTION' in pg.locator('output').inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
  page.goto(base+'index.html');page.locator('#lesson-nav a[href="lessons/'+S+'.html"]').wait_for()
  page.goto(base+'notebooks.html');page.locator('#lab-B02').wait_for()
  page.goto(base+'labs/html/'+S+'.html');assert 'COMPLETE_SELECTED_RELEASE_PROTOCOL' in page.locator('body').inner_text()
  assert page.locator('img').first.evaluate('(el)=>el.complete&&el.naturalWidth>0')
  page.locator('img').first.screenshot(path=str(V/'notebook-architecture.png'));browser.close()
finally:server.shutdown();server.server_close()
assert not errors,errors
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
links=0
for path in paths[:2]:
 parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
r=dict(status='PASS',desktop_mobile_selection_states=states,keyboard_reset='PASS',print_nojs='PASS',portable_source_parity='PASS',deterministic_builder='PASS',archive_replay='PASS',manifest_galleries='PASS',local_links=links,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_b02_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
