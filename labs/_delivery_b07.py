"""Portable fresh execution, live learner guards, browser behavior and geometry."""
import ast,functools,hashlib,http.server,json,os,subprocess,sys,tempfile,threading,zipfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';E=P/'evidence/b07';S='b07-semantic-transfer';V=R/'reviews/lesson-b07'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solpath=P/'solutions'/(S+'.ipynb');solution=nbformat.read(solpath,4)
sc='\n\n'.join(c.source for c in student.cells if c.cell_type=='code');code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert sc.count('NotImplementedError')==3 and 'NotImplementedError' not in code
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert json.loads((P/'_execution_b07_results.json').read_text())['executed_code_sha256']==hashlib.sha256(code.encode()).hexdigest()
funcs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for name in ['relkit/semantic_b07.py','relkit/carte_b07.py','_run_b07.py','_audit_b07.py','_source_b07.py']:
 for n in ast.parse((P/name).read_text()).body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)):assert funcs[n.name]==ast.dump(n,include_attributes=False),n.name
with tempfile.TemporaryDirectory(prefix='b07-student-') as td:
 try:NotebookClient(student,timeout=120,kernel_name='python3',resources={'metadata':{'path':td}}).execute()
 except CellExecutionError as exc:assert 'NotImplementedError' in str(exc) and 'TODO: intervene' in str(exc)
 else:raise AssertionError('Blank student unexpectedly passed')
paths=[R/'lessons'/(S+'.html'),R/'reference/b07-semantic-transfer.html',P/(S+'.ipynb'),E/'reproducer.zip']
before={p:p.read_bytes() for p in paths};saved=solpath.read_bytes()
try:
 subprocess.run([sys.executable,str(P/'_build_b07.py')],check=True,capture_output=True)
 rebuilt=nbformat.read(solpath,4);assert '\n\n'.join(c.source for c in rebuilt.cells if c.cell_type=='code')==code
 assert all(p.read_bytes()==b for p,b in before.items()),'Nondeterministic builder'
finally:solpath.write_bytes(saved)
with tempfile.TemporaryDirectory(prefix='b07-archive-') as td:
 with zipfile.ZipFile(E/'reproducer.zip') as z:z.extractall(td)
 lab=Path(td)/'labs';fresh=Path(td)/'fresh-runs'
 subprocess.run([sys.executable,str(lab/'_run_b07.py'),'--output',str(fresh)],cwd=td,check=True,capture_output=True,timeout=180)
 assert json.loads((fresh/'results.json').read_text())==json.loads((E/'runs/results.json').read_text()),'Fresh CLI fit result mismatch'
 result=subprocess.run([sys.executable,str(lab/'_reproduce_b07.py'),'--run'],cwd=td,capture_output=True,text=True)
 assert result.returncode!=0 and 'INCOMPLETE_SOURCE_PROTOCOL' in result.stderr
 target=lab/'sources/b07/original/contexttab/contexttab.py';target.write_bytes(target.read_bytes()+b'changed')
 result=subprocess.run([sys.executable,str(lab/'_reproduce_b07.py')],cwd=td,capture_output=True,text=True)
 assert result.returncode!=0 and 'Source artifact mismatch' in result.stderr
 target=lab/'data/b07/wine_pl.parquet';target.write_bytes(target.read_bytes()+b'changed')
 result=subprocess.run([sys.executable,str(lab/'_run_b07.py'),'--output',str(Path(td)/'corrupt-run')],cwd=td,capture_output=True,text=True)
 assert result.returncode!=0 and 'Input authentication failed' in result.stderr
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*a):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base='http://127.0.0.1:'+str(server.server_port)+'/'
errors=[];states=0;geometry=[]
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.on('response',lambda r:errors.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
  for width in [1200,375]:
   page.set_viewport_size({'width':width,'height':950});page.goto(base+'lessons/'+S+'.html');board=page.locator('[data-b07="intervention"]')
   for val,center,count in [('meaningful','1,0.5','2'),('anonymous','0.36,0.94','2'),('numeric_only','0.72,1.28','1')]:
    board.get_by_label('Information arm').select_option(val);assert board.locator('output').get_attribute('data-center')==center
    assert board.locator('output').get_attribute('data-features')==count;states+=1
   board.get_by_role('button').click();assert board.get_by_label('Information arm').input_value()=='meaningful'
   board.get_by_label('Information arm').focus();page.keyboard.press('ArrowDown');assert board.get_by_label('Information arm').input_value()=='anonymous';board.get_by_role('button').click()
   contract=page.locator('[data-b07="adaptation"]')
   for model in ['carte','contexttab','tabstar']:
    contract.get_by_label('Adaptation path').select_option(model)
    for target in ['all','answer']:
     contract.get_by_label('Candidate-class input').select_option(target);assert contract.locator('output').get_attribute('data-valid')==str(target=='all').lower();states+=1
   contract.get_by_role('button').click()
   prediction=page.locator('#b07-predict');assert prediction.locator('.predict-reveal').is_disabled();prediction.locator('.predict-option').first.click();assert prediction.locator('.predict-reveal').is_enabled();prediction.locator('.predict-reveal').click();assert 'wine_pl' in prediction.inner_text()
   tb=page.locator('#b07-teachback');assert tb.locator('button').first.is_disabled();tb.locator('textarea').fill('Renaming keeps values but removes natural headers. Text removal changes available facts. Every target candidate is included without revealing the actual answer. Context and adapter updates differ.');assert tb.locator('button').first.is_enabled();tb.locator('button').first.click();assert tb.locator('input[type=checkbox]').count()==4
   assert page.locator('#b07-warmup button').count()>0
   page.locator('#b07-results summary').click();assert page.locator('#b07-results table').count()==2
   assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),width
   for img in page.locator('article img').all():assert img.evaluate('(el)=>el.complete&&el.naturalWidth>0')
   page.evaluate('scrollTo(0,0)');page.screenshot(path=str(V/f'top-{width}.png'))
   board.screenshot(path=str(V/f'intervention-{width}.png'));contract.screenshot(path=str(V/f'adaptation-{width}.png'))
   for i,name in enumerate(['carte','contexttab','tabstar','results']):page.locator('.b07-figure').nth(i).screenshot(path=str(V/f'{name}-{width}.png'))
  page.emulate_media(media='print');assert board.locator('select').first.evaluate('(el)=>getComputedStyle(el).display')=='none';page.emulate_media(media='screen')
  context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});pg=context.new_page();pg.goto(base+'lessons/'+S+'.html');assert '[1,0.5]' in pg.locator('[data-b07="intervention"] output').inner_text();pg.locator('#b07-results summary').click();assert pg.locator('#b07-results table').count()==2;assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
  page.goto(base+'index.html');page.locator('#lesson-nav a[href="lessons/'+S+'.html"]').wait_for()
  page.goto(base+'notebooks.html');page.locator('#lab-B07').wait_for();assert page.locator('#nb-list li').first.get_attribute('id')=='lab-B07'
  page.goto(base+'labs/html/'+S+'.html');assert 'INCOMPLETE_SOURCE_PROTOCOL' in page.locator('body').inner_text();assert page.locator('img').count()==4
  for i in range(4):
   assert page.locator('img').nth(i).evaluate('(el)=>el.complete&&el.naturalWidth>0')
  page.set_viewport_size({'width':1000,'height':1100});page.locator('img').nth(2).screenshot(path=str(V/'notebook-tabstar.png'))
  for name in ['carte','contexttab','tabstar']:
   page.goto(base+'labs/figures/b07/'+name+'.svg')
   bad=page.evaluate('''()=>Array.from(document.querySelectorAll('text')).map(e=>({text:e.textContent,box:e.getBBox()})).filter(o=>o.box.x<0||o.box.y<0||o.box.x+o.box.width>460||o.box.y+o.box.height>document.querySelector('svg').viewBox.baseVal.height)''')
   assert not bad,(name,bad);geometry.append(name)
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
result=dict(status='PASS',student_blank='EXPECTED_TODO_FAILURE',notebook_source_parity='EXACT',deterministic_builder='PASS',standalone_fresh_cli='27fits and full predictions EXACT',source_data_corruption='REJECTED',browser_states=states,geometry=geometry,desktop_mobile=[1200,375],keyboard_reset='PASS',prediction_teachback_gates='PASS',print_nojs='PASS',manifest_navigation='PASS',local_links=links,live_colab='NOT_CHECKED')
(P/'_delivery_b07_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
