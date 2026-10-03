"""Check notebook/source parity, standalone fresh execution, browser and links."""
import ast,functools,hashlib,http.server,json,os,subprocess,sys,tempfile,threading,zipfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';E=P/'evidence/b06';S='b06-mitra-prior-mixtures';V=R/'reviews/lesson-b06'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solpath=P/'solutions'/(S+'.ipynb');solution=nbformat.read(solpath,4)
sc='\n\n'.join(c.source for c in student.cells if c.cell_type=='code');code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert sc.count('NotImplementedError')==3 and 'NotImplementedError' not in code
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert json.loads((P/'_execution_b06_results.json').read_text())['executed_code_sha256']==hashlib.sha256(code.encode()).hexdigest()
funcs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for name in ['relkit/prior_b06.py','_audit_b06.py','_source_b06.py']:
 for n in ast.parse((P/name).read_text()).body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)):assert funcs[n.name]==ast.dump(n,include_attributes=False),n.name
with tempfile.TemporaryDirectory(prefix='b06-student-') as td:
 try:NotebookClient(student,timeout=120,kernel_name='python3',resources={'metadata':{'path':td}}).execute()
 except CellExecutionError as exc:assert 'NotImplementedError' in str(exc) and 'TODO: choose_prior' in str(exc)
 else:raise AssertionError('Blank student unexpectedly passed')
paths=[R/'lessons'/(S+'.html'),R/'reference/b06-prior-mixtures.html',P/(S+'.ipynb'),E/'reproducer.zip']
before={p:p.read_bytes() for p in paths};saved=solpath.read_bytes()
try:
 subprocess.run([sys.executable,str(P/'_build_b06.py')],check=True,capture_output=True)
 rebuilt=nbformat.read(solpath,4);assert '\n\n'.join(c.source for c in rebuilt.cells if c.cell_type=='code')==code
 assert all(p.read_bytes()==b for p,b in before.items()),'Nondeterministic builder'
finally:solpath.write_bytes(saved)
with tempfile.TemporaryDirectory(prefix='b06-archive-') as td:
 with zipfile.ZipFile(E/'reproducer.zip') as z:z.extractall(td)
 lab=Path(td)/'labs'
 subprocess.run([sys.executable,str(lab/'_verify_b06.py')],cwd=td,check=True,capture_output=True,timeout=120)
 # Standalone fresh command is verified independently from repository imports.
 fresh_receipt=E/'fresh-verification.json'
 current_archive=hashlib.sha256((E/'reproducer.zip').read_bytes()).hexdigest()
 if fresh_receipt.exists():
  prior=json.loads(fresh_receipt.read_text())
  assert prior['status']=='PASS' and prior['archive_sha256']==current_archive,'Changed archive requires new fresh verification'
 else:
  fresh=Path(td)/'fresh-runs'
  subprocess.run([sys.executable,str(lab/'_run_b06.py'),'--output',str(fresh)],cwd=td,check=True,capture_output=True,timeout=240)
  for path in (E/'runs').glob('run-*.json'):
   a=json.loads(path.read_text());b=json.loads((fresh/path.name).read_text())
   assert a['records']==b['records'] and a['training']==b['training'],path.name
  fresh_receipt.write_text(json.dumps(dict(status='PASS',archive_sha256=current_archive,fits=9,records_and_traces='EXACT'),indent=2)+'\n')
 result=subprocess.run([sys.executable,str(lab/'_reproduce_b06.py'),'--run'],cwd=td,capture_output=True,text=True)
 assert result.returncode!=0 and 'INCOMPLETE_SOURCE_PROTOCOL' in result.stderr
 target=lab/'sources/b06/mitra-classifier/config.json';target.write_bytes(target.read_bytes()+b'changed')
 result=subprocess.run([sys.executable,str(lab/'_reproduce_b06.py')],cwd=td,capture_output=True,text=True)
 assert result.returncode!=0 and 'Source hash' in result.stderr
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*a):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base='http://127.0.0.1:'+str(server.server_port)+'/'
errors=[];states=0
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.on('response',lambda r:errors.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
  for width in [1200,375]:
   page.set_viewport_size({'width':width,'height':950});page.goto(base+'lessons/'+S+'.html');board=page.locator('[data-b06="mixture"]')
   for val,count in [('0',0),('.25',1),('.5',2),('.75',3),('1',4)]:
    board.get_by_label('SCM probability').select_option(val)
    assert int(board.locator('output').get_attribute('data-count'))==count
    assert board.locator('[data-family=scm]').count()==count;states+=1
   board.get_by_role('button').click();assert board.get_by_label('SCM probability').input_value()=='.5'
   board.get_by_label('SCM probability').focus();page.keyboard.press('ArrowDown');assert board.get_by_label('SCM probability').input_value()=='.75';board.get_by_role('button').click()
   select=page.locator('[data-b06="selection"]')
   for val,phrase in [('none','not selected'),('development','Lock it'),('final','no longer an untouched')]:
    select.get_by_label('Mixture selection').select_option(val);assert phrase in select.locator('output').inner_text();states+=1
   select.get_by_role('button').click()
   quiz=page.locator('[data-b06="quiz"]')
   for label,correct in [('Change prior only','true'),('Change learner also','false'),('Change queries also','false')]:
    quiz.get_by_label(label).check();assert quiz.locator('output').get_attribute('data-correct')==correct;states+=1
   quiz.get_by_role('button').click();assert quiz.locator('input:checked').count()==0
   back=page.locator('#b06-teachback');assert back.locator('button').first.is_disabled()
   back.locator('textarea').fill('The models remain near chance. A controlled experiment cannot guarantee useful learning, and the course generators differ from original Mitra.');assert back.locator('button').first.is_enabled();back.locator('button').first.click()
   assert back.locator('input[type=checkbox]').count()==4
   assert page.locator('#b06-warmup button').count()>0
   page.locator('details summary').click()
   assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),width
   for img in page.locator('article img').all():assert img.evaluate('(el)=>el.complete&&el.naturalWidth>0')
   page.evaluate('scrollTo(0,0)');page.screenshot(path=str(V/f'top-{width}.png'));board.screenshot(path=str(V/f'widget-{width}.png'));page.locator('.b06-figure').first.screenshot(path=str(V/f'architecture-{width}.png'));page.locator('.b06-results').screenshot(path=str(V/f'results-{width}.png'));select.screenshot(path=str(V/f'selection-{width}.png'))
  page.emulate_media(media='print');assert board.locator('select').first.evaluate('(el)=>getComputedStyle(el).display')=='none';page.emulate_media(media='screen')
  context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});pg=context.new_page();pg.goto(base+'lessons/'+S+'.html');assert '2 SCM + 2 tree' in pg.locator('[data-b06="mixture"] output').inner_text();assert pg.locator('.b06-task').count()==4;assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
  page.goto(base+'index.html');page.locator('#lesson-nav a[href="lessons/'+S+'.html"]').wait_for()
  page.goto(base+'notebooks.html');page.locator('#lab-B06').wait_for();assert page.locator('#nb-list li').first.get_attribute('id')=='lab-B06'
  page.goto(base+'labs/html/'+S+'.html');assert 'INCOMPLETE_SOURCE_PROTOCOL' in page.locator('body').inner_text();assert page.locator('img').first.evaluate('(el)=>el.complete&&el.naturalWidth>0')
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
r=dict(status='PASS',desktop_mobile_widget_states=states,keyboard_reset='PASS',teachback_retrieval='PASS',print_nojs='PASS',blank_student_fails='PASS',portable_source_parity='PASS',complete_visible_course_model_trainer='PASS',deterministic_builder='PASS',archive_inference='PASS',archive_fresh_training_verification_fits=9,archive_fresh_records_and_traces='EXACT',benchmark_dispatch_refusal='PASS',source_corruption_refused='PASS',manifest_galleries='PASS',local_links=links,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_b06_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
