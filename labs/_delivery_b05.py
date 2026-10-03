"""Portable execution, actual browser interactions, source parity and publication links."""
import ast,functools,hashlib,http.server,json,math,os,subprocess,sys,tempfile,threading,zipfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';E=P/'evidence/b05';S='b05-tabdpt-real-data-retrieval';V=R/'reviews/lesson-b05'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solpath=P/'solutions'/(S+'.ipynb');solution=nbformat.read(solpath,4)
sc='\n\n'.join(c.source for c in student.cells if c.cell_type=='code');code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert sc.count('NotImplementedError')==3 and 'NotImplementedError' not in code
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert json.loads((P/'_execution_b05_results.json').read_text())['executed_code_sha256']==hashlib.sha256(code.encode()).hexdigest()
funcs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
for name in ['relkit/retrieval_b05.py','_audit_b05.py','_source_b05.py']:
 for n in ast.parse((P/name).read_text()).body:
  if isinstance(n,ast.FunctionDef):assert funcs[n.name]==ast.dump(n,include_attributes=False),n.name
markdown='\n'.join(c.source for c in solution.cells if c.cell_type=='markdown')
for f in [*sorted((P/'sources/b05/original/src/tabdpt').glob('*.py')),P/'sources/b05/training/dataset.py',P/'sources/b05/training/transformer_layer.py',P/'_source_probe_b05.py']:
 assert f.read_text() in markdown,f
with tempfile.TemporaryDirectory(prefix='b05-student-') as td:
 try:NotebookClient(student,timeout=120,kernel_name='python3',resources={'metadata':{'path':td}}).execute()
 except CellExecutionError as exc:assert 'NotImplementedError' in str(exc) and 'TODO: feature_view' in str(exc)
 else:raise AssertionError('Blank student unexpectedly passed')
paths=[R/'lessons'/(S+'.html'),R/'reference/b05-retrieval-episodes.html',P/(S+'.ipynb'),E/'reproducer.zip']
before={p:p.read_bytes() for p in paths};saved=solpath.read_bytes()
try:
 subprocess.run([sys.executable,str(P/'_build_b05.py')],check=True,capture_output=True)
 rebuilt=nbformat.read(solpath,4);assert '\n\n'.join(c.source for c in rebuilt.cells if c.cell_type=='code')==code
 assert all(p.read_bytes()==b for p,b in before.items()),'Nondeterministic builder'
finally:solpath.write_bytes(saved)
with tempfile.TemporaryDirectory(prefix='b05-archive-') as td:
 with zipfile.ZipFile(E/'reproducer.zip') as z:z.extractall(td)
 subprocess.run([sys.executable,str(Path(td)/'labs/_verify_b05.py')],cwd=td,check=True,capture_output=True,timeout=120)
 result=subprocess.run([sys.executable,str(Path(td)/'labs/_reproduce_b05.py'),'--run'],cwd=td,capture_output=True,text=True)
 assert result.returncode!=0 and 'INCOMPLETE_SOURCE_PROTOCOL' in result.stderr
 target=Path(td)/'labs/sources/b05/training/dataset.py';target.write_bytes(target.read_bytes()+b'changed')
 result=subprocess.run([sys.executable,str(Path(td)/'labs/_reproduce_b05.py')],cwd=td,capture_output=True,text=True)
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
   page.set_viewport_size({'width':width,'height':950});page.goto(base+'lessons/'+S+'.html');board=page.locator('[data-b05="retrieval"]')
   for mode in ['excluded','included']:
    for target in ['original','changed']:
     for k in [2,3,4]:
      board.get_by_label('Distance features').select_option(mode);board.get_by_label('Target values').select_option(target);board.get_by_label('Neighborhood size').select_option(str(k))
      x=[0,.1,.2,.3,1,2];y=[0,100,0,100,0,100] if target=='original' else [0,0,100,0,100,100]
      def var(a):return sum((v-sum(a)/len(a))**2 for v in a)/len(a)
      ds=[v*v/var(x)+((y[i]-y[0])**2/var(y) if mode=='included' else 0) for i,v in enumerate(x)]
      expected=sorted(range(6),key=lambda i:(ds[i],i))[:k]
      assert json.loads(board.locator('output').get_attribute('data-ids'))==expected
      for i,cell in enumerate(board.locator('tbody tr td:last-child').all()):assert abs(float(cell.inner_text())-ds[i])<=.000051
      assert board.locator('tr[data-picked=true]').count()==k;states+=1
   board.get_by_role('button').click();assert board.get_by_label('Neighborhood size').input_value()=='3'
   board.get_by_label('Neighborhood size').focus();page.keyboard.press('ArrowDown');assert board.get_by_label('Neighborhood size').input_value()=='4';board.get_by_role('button').click()
   quiz=page.locator('[data-b05="quiz"]')
   for label,correct in [('Targets enter retrieval','false'),('Targets leave retrieval','true'),('Targets become predictions','false')]:
    quiz.get_by_label(label).check();assert quiz.locator('output').get_attribute('data-correct')==correct
   quiz.get_by_role('button').click();assert quiz.locator('input:checked').count()==0
   assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),width
   assert page.locator('.b05-figure img').evaluate('(el)=>el.complete&&el.naturalWidth>0')
   page.evaluate('scrollTo(0,0)');page.screenshot(path=str(V/f'top-{width}.png'));board.screenshot(path=str(V/f'widget-{width}.png'));page.locator('.b05-figure').screenshot(path=str(V/f'architecture-{width}.png'))
  page.emulate_media(media='print');assert board.locator('select').first.evaluate('(el)=>getComputedStyle(el).display')=='none';page.emulate_media(media='screen')
  context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});pg=context.new_page();pg.goto(base+'lessons/'+S+'.html');assert '[0, 1, 2]' in pg.locator('[data-b05="retrieval"] output').inner_text();assert pg.locator('[data-b05="retrieval"] tbody tr').count()==6;assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
  page.goto(base+'index.html');page.locator('#lesson-nav a[href="lessons/'+S+'.html"]').wait_for()
  page.goto(base+'notebooks.html');page.locator('#lab-B05').wait_for();assert page.locator('#nb-list li').first.get_attribute('id')=='lab-B05'
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
r=dict(status='PASS',desktop_mobile_retrieval_states=states,quiz_states=6,keyboard_reset='PASS',print_nojs='PASS',blank_student_fails='PASS',portable_source_parity='PASS',complete_visible_model_source='PASS',deterministic_builder='PASS',archive_execution='PASS',benchmark_dispatch_refusal='PASS',source_corruption_refused='PASS',manifest_galleries='PASS',local_links=links,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_b05_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
