"""Notebook identity, browser behavior and copied Pages delivery verification."""
from _gallery_delivery import reveal_gallery_link
import ast,functools,hashlib,json,os,re,subprocess,sys,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0130-rdl-checkpoint'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
student=nbformat.read(P/f'{S}.ipynb',as_version=4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert not any(c.outputs for c in student.cells if c.cell_type=='code')
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert 'from relkit' not in code and '__file__' not in code
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_l130_results.json').read_text())['executed_code_sha256']
def definitions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
canonical=definitions((P/'relkit/checkpoint_l130.py').read_text());canonical.update(definitions((P/'relkit/rdl_l117.py').read_text()));canonical.update(definitions((P/'relkit/batch_audit_l123.py').read_text()))
sol=definitions(code);stu=definitions('\n\n'.join(c.source for c in student.cells if c.cell_type=='code'))
for name,node in canonical.items():
 assert sol[name]==node,name
 if name not in ['validate_queries','keyed_mae','reproduction_verdict']:assert stu[name]==node,name
paths=[R/'lessons'/f'{S}.html',R/'reference/rdl-checkpoint.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb'];before=[sha(p) for p in paths]
subprocess.run([sys.executable,str(P/'_build_l130.py')],check=True,capture_output=True);subprocess.run([sys.executable,str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True);assert before==[sha(p) for p in paths],'Builder drift'
figs=sorted((P/'figures/l130').glob('*'));before=[sha(p) for p in figs]
subprocess.run([sys.executable,str(P/'_figures_l130.py')],check=True,capture_output=True);assert before==[sha(p) for p in figs],'Figure drift'
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri())
  keyed=page.locator('#l130-keyed');key_select=keyed.locator('select');key_out=keyed.locator('output')
  assert 'Positional MAE: 0.5' in key_out.inner_text() and 'Keyed MAE: 0.5' in key_out.inner_text()
  key_select.select_option('reversed');assert 'Positional MAE: 2.5' in key_out.inner_text() and 'Keyed MAE: 0.5' in key_out.inner_text();states+=1
  key_select.select_option('duplicate');assert 'REJECTED' in key_out.inner_text();states+=1
  keyed.get_by_role('button',name='Reset',exact=True).click();assert key_select.input_value()=='ordered'
  key_select.focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert key_select.input_value()=='reversed'
  keyed.screenshot(path=f'/tmp/l130-keyed-{width}.png')
  w=page.locator('#l130-checkpoint');control=w.locator('select');out=w.locator('output');slider=w.locator('input')
  assert 'Epoch 2 selected' in out.inner_text()
  for value in ['0','4','10']:
   slider.fill(value);slider.dispatch_event('input');assert 'Epoch 2 selected' in out.inner_text();states+=1
  slider.fill('1');slider.dispatch_event('input');control.select_option('test')
  assert 'Epoch 3 selected' in out.inner_text() and 'Invalid' in out.inner_text();states+=1
  w.get_by_role('button',name='Reset',exact=True).click();assert control.input_value()=='val' and slider.input_value()=='1'
  control.focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert control.input_value()=='test'
  w.get_by_role('button',name='Reset',exact=True).click()
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),'Page overflow'
  assert page.locator('figure').count()==5
  page.locator('figure img').evaluate_all('(xs)=>xs.forEach(x=>x.loading="eager")');page.wait_for_function('Array.from(document.querySelectorAll("figure img")).every(x=>x.complete && x.naturalWidth>0)');assert page.locator('figure img').evaluate_all('(xs)=>xs.every(x=>x.complete && x.naturalWidth>0)')
  page.screenshot(path=f'/tmp/l130-page-{width}.png')
  w.screenshot(path=f'/tmp/l130-cutoff-{width}.png')
  for i in range(5):page.locator('figure').nth(i).screenshot(path=f'/tmp/l130-figure-{i}-{width}.png')
 page.set_viewport_size({'width':1000,'height':900});page.goto((P/'html'/f'{S}.html').as_uri())
 assert page.locator('figure img[src^="data:image/png;base64,"]').count()==5
 page.locator('figure').nth(0).screenshot(path='/tmp/l130-notebook-collision.png')
 nojs=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});plain=nojs.new_page();plain.goto((R/'lessons'/f'{S}.html').as_uri())
 assert 'Static trace:' in plain.inner_text('article') and 'PENDING_WRITTEN_DEFENSE' in plain.inner_text('article')
 assert plain.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
 plain.set_viewport_size({'width':900,'height':1000});plain.emulate_media(media='print')
 assert plain.locator('figure').count()==5
 assert plain.locator('figure img').evaluate_all('(xs)=>xs.every(x=>getComputedStyle(x).minWidth==="0px")')
 plain.screenshot(path='/tmp/l130-print.png');nojs.close();browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
(P/'_delivery_l130_results.json').write_text('{"status":"RUNNING"}\n')
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))];checked=0
with tempfile.TemporaryDirectory(prefix='l130-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/rdl-checkpoint.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/checkpoint_l130.py','labs/relkit/rdl_l117.py','labs/_run_l117.py','labs/l130-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l130/summary.json','labs/evidence/l130/trace.json','labs/evidence/l130/comparison.json','labs/l130-defense.md','labs/relkit/batch_audit_l123.py','labs/_run_l130.py','labs/sources/l117/model.py','modal/l130_repro.py']:
  assert (stage/rel).exists(),rel
 assert not (stage/'labs/results/l130').exists(),'Do not copy checkpoint payloads'
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));base=f'http://127.0.0.1:{server.server_port}'
   page.goto(base+'/index.html');reveal_gallery_link(page, 'a[href="lessons/'+S+'.html"]');assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(base+'/notebooks.html');reveal_gallery_link(page, 'a[href="labs/html/'+S+'.html"]')
   page.goto(base+'/lessons/'+S+'.html');assert 'Epoch 2 selected' in page.locator('#l130-checkpoint output').inner_text();browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
r={'status':'PASS','browser_widths':[1200,375],'interactive_states':states,'keyboard_reset':'PASS','print_nojs':'PASS',
   'portable_figures':5,'live_tasks':3,'canonical_definitions':len(canonical),'executed_code_hash':'MATCH','deterministic_rebuild':'EXACT',
   'copied_pages_links':checked,'javascript_errors':errors,'screenshots':'/tmp/l130-*.png','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/'_delivery_l130_results.json').write_text(json.dumps(r,indent=2));print(r)
