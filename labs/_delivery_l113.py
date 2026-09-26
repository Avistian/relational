"""Notebook/source alignment, browser interactions and actual copied Pages build."""
import ast,functools,hashlib,json,os,re,subprocess,sys,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0113-scaling-ogb'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
student=nbformat.read(P/f'{S}.ipynb',as_version=4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert not any(c.outputs for c in student.cells if c.cell_type=='code')
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert 'from relkit' not in code and '__file__' not in code
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_l113_results.json').read_text())['executed_code_sha256']
def definitions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
canonical=definitions((P/'relkit/scaling_l113.py').read_text());sol=definitions(code)
for name,node in canonical.items():assert sol[name]==node,name
paths=[R/'lessons'/f'{S}.html',R/'reference/scaling-ogb.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb'];before=[sha(p) for p in paths]
subprocess.run([sys.executable,str(P/'_build_l113.py')],check=True,capture_output=True);assert before==[sha(p) for p in paths],'Builder drift'
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('requestfailed',lambda r:errors.append(r.url))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri());widget=page.locator('[data-scaling-trace]');select=widget.locator('select')
  for value,expect in [('1','Output = 6'),('2','Output = 9'),('3','Output = 9')]:
   select.select_option(value);assert expect in widget.inner_text();states+=1
  widget.locator('button').click();assert select.input_value()=='1';select.focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert select.input_value()=='2';widget.locator('button').click()
  ranges=page.locator('[data-scaling-memory] input')
  for roots in [64,256,1024]:
   for fanout in [2,10,20]:
    ranges.nth(0).fill(str(roots));ranges.nth(1).fill(str(fanout));n=roots*(1+fanout+fanout**2+fanout**3);assert f'{n:,} node occurrences' in page.locator('[data-scaling-memory]').inner_text();states+=1
  ranges.nth(0).fill('256');ranges.nth(1).fill('10')
  tb=page.locator('#l113-teachback');assert tb.locator('button').first.is_disabled();tb.locator('textarea').fill('Finish every layer before the next layer reads its completed features.');tb.locator('button').first.click();assert 'Every incoming neighbor' in tb.inner_text()
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),'Page overflow'
  assert page.locator('figure').count()==5
  for figure in page.locator('figure').all():
   assert figure.locator('img').evaluate('(e)=>e.complete&&e.naturalWidth>0')
  page.screenshot(path=f'/tmp/l113-page-{width}.png',full_page=True);widget.screenshot(path=f'/tmp/l113-widget-{width}.png')
  page.locator('[data-scaling-memory]').screenshot(path=f'/tmp/l113-memory-{width}.png')
 page.set_viewport_size({'width':1000,'height':900})
 page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('img[src^="data:image/png;base64,"]').count()==5
 for i in range(5):page.locator('img[src^="data:image/png;base64,"]').nth(i).screenshot(path=f'/tmp/l113-notebook-{i}.png')
 nojs=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});plain=nojs.new_page();plain.goto((R/'lessons'/f'{S}.html').as_uri());assert plain.locator('figure').count()==5;assert plain.evaluate('document.documentElement.scrollWidth<=innerWidth+1');plain.emulate_media(media='print');assert plain.locator('details p').first.evaluate('(e)=>e.checkVisibility()');plain.screenshot(path='/tmp/l113-print.png',full_page=True);nojs.close();browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0];lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))];checked=0
with tempfile.TemporaryDirectory(prefix='l113-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines);subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/scaling-ogb.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/scaling_l113.py','labs/_run_l113.py','labs/l113-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l113/summary.json','labs/sources/l113/cluster_gcn.py','modal/l113_repro.py']:assert (stage/rel).exists(),rel
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));base=f'http://127.0.0.1:{server.server_port}'
   page.goto(base+'/index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]');assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(base+'/notebooks.html');page.wait_for_selector('a[href="labs/html/'+S+'.html"]')
   page.goto(base+'/lessons/'+S+'.html');assert 'Output = 6' in page.locator('[data-scaling-trace]').inner_text();browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
r={'status':'PASS','browser_widths':[1200,375],'widget_states':states,'keyboard_reset':'PASS','print_nojs':'PASS','portable_figures':5,'live_student_tasks':3,'canonical_definitions':len(canonical),'executed_code_hash':'MATCH','deterministic_rebuild':'EXACT','copied_pages_links':checked,'javascript_errors':errors,'screenshots':'/tmp/l113-*.png','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/'_delivery_l113_results.json').write_text(json.dumps(r,indent=2));print(r)
