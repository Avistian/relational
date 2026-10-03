"""Portable evidence, deterministic builds, browser/mobile interaction and link checks."""
import ast,functools,hashlib,http.server,json,os,subprocess,sys,tempfile,threading,zipfile,tarfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';E=P/'evidence/b04a';S='b04a-scaling-rows-features-classes';V=R/'reviews/lesson-b04a';V.mkdir(exist_ok=True)
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution_path=P/'solutions'/(S+'.ipynb');solution=nbformat.read(solution_path,4)
sc='\n\n'.join(c.source for c in student.cells if c.cell_type=='code');code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert sc.count('NotImplementedError')==3 and 'NotImplementedError' not in code
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert json.loads((P/'_execution_b04a_results.json').read_text())['executed_code_sha256']==hashlib.sha256(code.encode()).hexdigest()
functions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
for name in ['relkit/scaling_b04a.py','_audit_b04a.py','_source_b04a.py','_verify_b04a.py','_test_b04a.py']:
 for n in ast.parse((P/name).read_text()).body:
  if isinstance(n,ast.FunctionDef):assert functions[n.name]==ast.dump(n,include_attributes=False),n.name
markdown='\n'.join(c.source for c in solution.cells if c.cell_type=='markdown')
for name in ['tabflex_model.py','linear_attention.py','tabflex_wrapper.py','tabpfn_wrapper.py']:
 assert (P/'sources/b04a'/name).read_text() in markdown,name
with tarfile.open(P/'sources/b04a/code.tar.gz') as tar:
 for archive,name in [('ticl/models/tabflex.py','tabflex_model.py'),('ticl/models/linear_attention.py','linear_attention.py'),('ticl/prediction/tabflex.py','tabflex_wrapper.py'),('ticl/prediction/tabpfn.py','tabpfn_wrapper.py')]:assert tar.extractfile(archive).read()==(P/'sources/b04a'/name).read_bytes()
paths=[R/'lessons'/(S+'.html'),R/'reference/b04a-scaling-axes.html',P/(S+'.ipynb'),E/'reproducer.zip']
before={p:p.read_bytes() for p in paths};executed=solution_path.read_bytes()
try:
 subprocess.run([sys.executable,str(P/'_build_b04a.py')],check=True,capture_output=True)
 rebuilt=nbformat.read(solution_path,4);assert '\n\n'.join(c.source for c in rebuilt.cells if c.cell_type=='code')==code
 assert all(p.read_bytes()==b for p,b in before.items()),'Nondeterministic build'
finally:solution_path.write_bytes(executed)
with tempfile.TemporaryDirectory(prefix='b04a-archive-') as td:
 with zipfile.ZipFile(E/'reproducer.zip') as z:z.extractall(td)
 subprocess.run([sys.executable,str(Path(td)/'labs/_verify_b04a.py')],cwd=td,check=True,capture_output=True,timeout=120)
 result=subprocess.run([sys.executable,str(Path(td)/'labs/_reproduce_b04a.py'),'--run'],cwd=td,capture_output=True,text=True)
 assert result.returncode!=0 and 'INCOMPLETE_SOURCE_PROTOCOL' in result.stderr
 result=subprocess.run([sys.executable,str(Path(td)/'labs/_run_b04a.py')],cwd=td,capture_output=True,text=True)
 assert result.returncode==0 and 'COMPLETE 108' in result.stdout
 target=Path(td)/'labs/relkit/scaling_b04a.py';target.write_bytes(target.read_bytes()+b'\n# changed helper\n')
 result=subprocess.run([sys.executable,str(Path(td)/'labs/_run_b04a.py')],cwd=td,capture_output=True,text=True)
 assert result.returncode!=0 and 'Changed execution' in result.stderr
 target=Path(td)/'labs/sources/b04a/linear_attention.py';target.write_bytes(target.read_bytes()+b'changed')
 result=subprocess.run([sys.executable,str(Path(td)/'labs/_reproduce_b04a.py')],cwd=td,capture_output=True,text=True)
 assert result.returncode!=0 and 'hash mismatch' in result.stderr
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
  report=json.loads((E/'audit.json').read_text())
  for width in [1200,375]:
   page.set_viewport_size({'width':width,'height':950});page.goto(base+'lessons/'+S+'.html');board=page.locator('[data-b04a="scaling"]')
   for s in [64,128,1024,32768]:
    for f in [8,64,256,1000]:
     for c in [2,10,11]:
      board.get_by_label('Support rows S').select_option(str(s));board.get_by_label('Raw features F').select_option(str(f));board.get_by_label('Classes C').select_option(str(c))
      out=board.locator('output');assert int(out.get_attribute('data-pairs'))==64*s;assert int(out.get_attribute('data-summary'))==16*c+16;assert int(out.get_attribute('data-projection'))==(s+64)*f*16;assert ('REJECT' in out.inner_text())==(c>10);states+=1
   board.get_by_role('button').click();assert board.get_by_label('Support rows S').input_value()=='128'
   board.get_by_label('Support rows S').focus();page.keyboard.press('ArrowDown');assert board.get_by_label('Support rows S').input_value()=='1024';board.get_by_role('button').click()
   measured=page.locator('[data-b04a="measured"]')
   for i,row in enumerate(report['rows']):
    measured.get_by_label('Course operating point').select_option(str(i));assert f"{row['log_loss']:.6f}" in measured.locator('output').inner_text();assert measured.locator('output').get_attribute('data-name')==row['name'],(i,row['name'],measured.locator('output').get_attribute('data-name'),measured.locator('select').input_value());measured_states+=1
   measured.get_by_role('button').click();assert measured.locator('select').input_value()=='0'
   quiz=page.locator('[data-b04a="quiz"]')
   for label,correct in [('Only memory changed','false'),('Prediction task changed','true'),('Nothing meaningful changed','false')]:
    quiz.get_by_label(label).check();assert quiz.locator('output').get_attribute('data-correct')==correct
   quiz.get_by_role('button').click();assert quiz.locator('input:checked').count()==0
   assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),width
   assert page.locator('.b04a-architecture img').evaluate('(el)=>el.complete&&el.naturalWidth>0')
   page.evaluate('scrollTo(0,0)');page.screenshot(path=str(V/f'top-{width}.png'));board.screenshot(path=str(V/f'axes-{width}.png'));page.locator('.b04a-architecture').screenshot(path=str(V/f'architecture-{width}.png'));measured.screenshot(path=str(V/f'evidence-{width}.png'))
  page.emulate_media(media='print');assert board.locator('select').first.evaluate('(el)=>getComputedStyle(el).display')=='none';page.emulate_media(media='screen')
  context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});pg=context.new_page();pg.goto(base+'lessons/'+S+'.html');assert '8,192' in pg.locator('[data-b04a="scaling"] output').inner_text();assert pg.locator('tbody tr').count()==15;assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
  page.goto(base+'index.html');page.locator('#lesson-nav a[href="lessons/'+S+'.html"]').wait_for()
  page.goto(base+'notebooks.html');page.locator('#lab-B04a').wait_for();assert page.locator('#nb-list li').first.get_attribute('id')=='lab-B04a'
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
r=dict(status='PASS',desktop_mobile_analytical_states=states,measured_states=measured_states,quiz_states=6,keyboard_reset='PASS',print_nojs='PASS',portable_source_parity='PASS',complete_visible_model_source='PASS',deterministic_builder='PASS',archive_execution='PASS',unchanged_resume='PASS',changed_helper_refused='PASS',benchmark_dispatch_refusal='PASS',source_corruption_refused='PASS',manifest_galleries='PASS',local_links=links,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_b04a_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
