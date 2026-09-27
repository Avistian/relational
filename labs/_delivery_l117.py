"""Standalone/canonical identity plus browser and realistic copied Pages checks."""
import ast,functools,hashlib,json,os,re,subprocess,sys,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0117-rdl-bridge'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
student=nbformat.read(P/f'{S}.ipynb',as_version=4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert not any(c.outputs for c in student.cells if c.cell_type=='code')
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert 'from relkit' not in code and '__file__' not in code
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_l117_results.json').read_text())['executed_code_sha256']
def definitions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
canonical=definitions((P/'relkit/rdl_l117.py').read_text());sol=definitions(code);stu=definitions('\n\n'.join(c.source for c in student.cells if c.cell_type=='code'))
for name,node in canonical.items():
 assert sol[name]==node,name
 if name not in ['foreign_key_edges','temporal_nodes','query_targets']:assert stu[name]==node,name
paths=[R/'lessons'/f'{S}.html',R/'reference/rdl-bridge.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb'];before=[sha(p) for p in paths];subprocess.run([sys.executable,str(P/'_build_l117.py')],check=True,capture_output=True);assert before==[sha(p) for p in paths],'Builder drift'
figs=sorted((P/'figures/l117').glob('*'));before=[sha(p) for p in figs];subprocess.run([sys.executable,str(P/'_figures_l117.py')],check=True,capture_output=True);assert before==[sha(p) for p in figs],'Figure drift'
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri())
  w=page.locator('#l117-cutoff');t=w.locator('[data-time]');h=w.locator('[data-hops]');enforce=w.locator('[data-enforce]');out=w.locator('output')
  assert '3 nodes; 0 future rows' in out.inner_text()
  t.fill('11');assert '4 nodes; 0 future rows' in out.inner_text()
  t.fill('7');enforce.uncheck();assert '4 nodes; 1 future rows' in out.inner_text()
  h.select_option('0');assert '1 nodes; 0 future rows' in out.inner_text()
  w.get_by_role('button',name='Reset trace').click();assert t.input_value()=='7' and enforce.is_checked() and h.input_value()=='2'
  enforce.focus();page.keyboard.press('Space');assert not enforce.is_checked();w.get_by_role('button',name='Reset trace').click();states+=5
  for cutoff in range(13):
   for hops in range(3):
    for filtered in [True,False]:
     ids={0}
     if hops>=1:ids.add(1)
     if hops>=2:
      if not filtered or cutoff>=5:ids.add(2)
      if not filtered or cutoff>=11:ids.add(3)
     actual=page.evaluate('(x)=>REGQueryViz.compute(...x)',[cutoff,hops,filtered]);assert actual==sorted(ids)
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),'Page overflow'
  assert page.locator('figure').count()==3
  page.screenshot(path=f'/tmp/l117-page-{width}.png',full_page=True);w.screenshot(path=f'/tmp/l117-widget-{width}.png');page.locator('figure').last.screenshot(path=f'/tmp/l117-architecture-{width}.png')
 page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('figure img[src^="data:image/png;base64,"]').count()==3
 page.locator('figure').last.screenshot(path='/tmp/l117-notebook-architecture.png')
 nojs=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});plain=nojs.new_page();plain.goto((R/'lessons'/f'{S}.html').as_uri());assert '4.13392' in plain.inner_text('article') and 'Static trace:' in plain.inner_text('article');assert plain.evaluate('document.documentElement.scrollWidth<=innerWidth+1');plain.emulate_media(media='print');assert plain.locator('details p').first.evaluate('(e)=>e.checkVisibility()');nojs.close();browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
(P/'_delivery_l117_results.json').write_text('{"status":"RUNNING"}\n')
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0];lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))];checked=0
with tempfile.TemporaryDirectory(prefix='l117-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines);subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/rdl-bridge.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/rdl_l117.py','labs/_run_l117.py','labs/l117-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l117/summary.json','labs/sources/l117/model.py','modal/l117_repro.py']:assert (stage/rel).exists(),rel
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));base=f'http://127.0.0.1:{server.server_port}'
   page.goto(base+'/index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]');assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(base+'/notebooks.html');page.wait_for_selector('a[href="labs/html/'+S+'.html"]')
   page.goto(base+'/lessons/'+S+'.html');assert '3 nodes' in page.locator('#l117-cutoff output').inner_text();browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
r={'status':'PASS','browser_widths':[1200,375],'interactive_states':states,'independent_widget_cases':156,'keyboard_reset':'PASS','print_nojs':'PASS','portable_figures':3,'live_tasks':3,'canonical_definitions':len(canonical),'executed_code_hash':'MATCH','deterministic_rebuild':'EXACT','copied_pages_links':checked,'javascript_errors':errors,'screenshots':'/tmp/l117-*.png','live_colab':'NOT_CHECKED','deployment':'Separate publication report'}
(P/'_delivery_l117_results.json').write_text(json.dumps(r,indent=2));print(r)
