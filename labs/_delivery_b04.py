"""Actual browser, portable archive, notebook parity and deterministic-build checks."""
import ast,functools,hashlib,http.server,json,math,os,subprocess,sys,tempfile,threading,zipfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';E=P/'evidence/b04';S='b04-tabicl-scalable-icl';V=R/'reviews/lesson-b04';V.mkdir(exist_ok=True)
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution_path=P/'solutions'/(S+'.ipynb');solution=nbformat.read(solution_path,4)
sc='\n\n'.join(c.source for c in student.cells if c.cell_type=='code');code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert sc.count('NotImplementedError')==3 and 'NotImplementedError' not in code
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert json.loads((P/'_execution_b04_results.json').read_text())['executed_code_sha256']==hashlib.sha256(code.encode()).hexdigest()
functions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
for name in ['relkit/scalable_b04.py','_audit_b04.py','_source_b04.py','_test_b04.py']:
 for n in ast.parse((P/name).read_text()).body:
  if isinstance(n,ast.FunctionDef):assert functions[n.name]==ast.dump(n,include_attributes=False),n.name
markdown='\n'.join(c.source for c in solution.cells if c.cell_type=='markdown')
up=P/'sources/b04/upstream'
for path in [*sorted((up/'src/tabicl/_model').glob('*.py')),up/'src/tabicl/_sklearn/classifier.py',up/'src/tabicl/_sklearn/preprocessing.py']:
 assert path.read_text() in markdown,path
paths=[R/'lessons'/(S+'.html'),R/'reference/b04-scalable-icl.html',P/(S+'.ipynb'),E/'reproducer.zip']
before={p:p.read_bytes() for p in paths};executed=solution_path.read_bytes()
try:
 subprocess.run([sys.executable,str(P/'_build_b04.py')],check=True,capture_output=True)
 rebuilt=nbformat.read(solution_path,4);assert '\n\n'.join(c.source for c in rebuilt.cells if c.cell_type=='code')==code
 assert all(p.read_bytes()==b for p,b in before.items()),'Non-deterministic build'
finally:solution_path.write_bytes(executed)
with tempfile.TemporaryDirectory(prefix='b04-archive-') as td:
 with zipfile.ZipFile(E/'reproducer.zip') as z:z.extractall(td)
 subprocess.run([sys.executable,str(Path(td)/'labs/_verify_b04.py')],cwd=td,check=True,capture_output=True,timeout=120)
 result=subprocess.run([sys.executable,str(Path(td)/'labs/_reproduce_b04.py'),'--run'],cwd=td,capture_output=True,text=True)
 assert result.returncode!=0 and 'INCOMPLETE_SOURCE_PROTOCOL' in result.stderr
 # A corrupted frozen source asset must not pass preflight.
 target=Path(td)/'labs/sources/b04/figure3a.svg';target.write_bytes(target.read_bytes()+b'changed')
 result=subprocess.run([sys.executable,str(Path(td)/'labs/_reproduce_b04.py')],cwd=td,capture_output=True,text=True)
 assert result.returncode!=0 and 'figure3a.svg' in result.stderr
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*a):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)))
threading.Thread(target=server.serve_forever,daemon=True).start();base='http://127.0.0.1:'+str(server.server_port)+'/'
errors=[];states=0;measured_states=0
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.on('response',lambda r:errors.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
  report=json.loads((E/'diagnostic-audit.json').read_text())
  for width in [1200,375]:
   page.set_viewport_size({'width':width,'height':950});page.goto(base+'lessons/'+S+'.html');board=page.locator('[data-b04="toy"]')
   for n in [2,8,64,256,1024,15001]:
    for gap in [0,1,2,4]:
     for mode in ['fixed','log']:
      board.get_by_label('Support keys').select_option(str(n));board.get_by_label('Anchor logit gap').select_option(str(gap));board.get_by_label('Scaling rule').select_option(mode)
      scale=math.log(n) if mode=='log' else 1;p=1/(1+(n-1)*math.exp(-gap*scale));other=(1-p)/(n-1)
      entropy=-(p*math.log(p)+(n-1)*other*math.log(other))/math.log(n)
      out=board.locator('output');assert abs(float(out.get_attribute('data-anchor'))-p)<1e-12;assert abs(float(out.get_attribute('data-entropy'))-entropy)<1e-12;states+=1
   board.get_by_role('button').click();assert board.get_by_label('Support keys').input_value()=='64'
   board.get_by_label('Support keys').focus();page.keyboard.press('ArrowDown');assert board.get_by_label('Support keys').input_value()=='256';board.get_by_role('button').click()
   measured=page.locator('[data-b04="measured"]')
   for i,row in enumerate(report['rows']):
    measured.get_by_label('Diagnostic configuration').select_option(str(i));assert f"{row['log_loss']:.6f}" in measured.locator('output').inner_text();assert measured.locator('output').get_attribute('data-name')==row['name'];measured_states+=1
   measured.get_by_role('button').click();assert measured.locator('select').input_value()=='1'
   quiz=page.locator('[data-b04="quiz"]')
   for label,correct in [('Accuracy must increase','false'),('Accuracy may change','true'),('Weights must update','false')]:
    quiz.get_by_label(label).check();assert quiz.locator('output').get_attribute('data-correct')==correct
   quiz.get_by_role('button').click();assert quiz.locator('input:checked').count()==0
   assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),width
   assert page.locator('.b04-architecture img').evaluate('(el)=>el.complete&&el.naturalWidth>0')
   page.evaluate('scrollTo(0,0)');page.screenshot(path=str(V/f'top-{width}.png'));board.screenshot(path=str(V/f'toy-{width}.png'));page.locator('.b04-architecture').screenshot(path=str(V/f'architecture-{width}.png'));measured.screenshot(path=str(V/f'evidence-{width}.png'))
  page.emulate_media(media='print');assert board.locator('select').first.evaluate('(el)=>getComputedStyle(el).display')=='none';page.emulate_media(media='screen')
  context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});pg=context.new_page();pg.goto(base+'lessons/'+S+'.html');assert '0.104975' in pg.locator('[data-b04="toy"] output').inner_text();assert pg.locator('tbody tr').count()==6;assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
  page.goto(base+'index.html');page.locator('#lesson-nav a[href="lessons/'+S+'.html"]').wait_for()
  page.goto(base+'notebooks.html');page.locator('#lab-B04').wait_for();assert page.locator('#nb-list li').first.get_attribute('id')=='lab-B04'
  page.goto(base+'labs/html/'+S+'.html');assert 'INCOMPLETE_SOURCE_PROTOCOL' in page.locator('body').inner_text();assert page.locator('img').first.evaluate('(el)=>el.complete&&el.naturalWidth>0');page.locator('img').first.screenshot(path=str(V/'notebook-architecture.png'))
  browser.close()
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
r=dict(status='PASS',desktop_mobile_analytical_states=states,measured_states=measured_states,quiz_states=6,keyboard_reset='PASS',print_nojs='PASS',portable_source_parity='PASS',complete_visible_model_source='PASS',deterministic_builder='PASS',archive_execution='PASS',benchmark_dispatch_refusal='PASS',source_corruption_refused='PASS',manifest_galleries='PASS',local_links=links,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_b04_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
