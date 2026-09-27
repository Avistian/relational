"""Standalone/canonical identity plus browser and realistic copied Pages checks."""
from _gallery_delivery import reveal_gallery_link
import ast,functools,hashlib,json,os,re,subprocess,sys,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0118-cvitkovic-relational-gnn'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
student=nbformat.read(P/f'{S}.ipynb',as_version=4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert not any(c.outputs for c in student.cells if c.cell_type=='code')
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert 'from relkit' not in code and '__file__' not in code
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_l118_results.json').read_text())['executed_code_sha256']
def definitions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
canonical=definitions((P/'relkit/cvitkovic_l118.py').read_text());sol=definitions(code);stu=definitions('\n\n'.join(c.source for c in student.cells if c.cell_type=='code'))
for name,node in canonical.items():
 assert sol[name]==node,name
 if name not in ['rdb_to_graph','normalized_sum','attention_pool']:assert stu[name]==node,name
paths=[R/'lessons'/f'{S}.html',R/'reference/cvitkovic-relational-gnn.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb'];before=[sha(p) for p in paths];subprocess.run([sys.executable,str(P/'_build_l118.py')],check=True,capture_output=True);assert before==[sha(p) for p in paths],'Builder drift'
figs=sorted((P/'figures/l118').glob('*'));before=[sha(p) for p in figs];subprocess.run([sys.executable,str(P/'_figures_l118.py')],check=True,capture_output=True);assert before==[sha(p) for p in figs],'Figure drift'
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri())
  w=page.locator('#l118-extraction');t=w.locator('[data-time]');mode=w.locator('[data-mode]');enforce=w.locator('[data-filter]');out=w.locator('output')
  assert out.inner_text().startswith('4 nodes:')
  mode.select_option('radius');assert out.inner_text().startswith('5 nodes:')
  enforce.check();assert out.inner_text().startswith('4 nodes:')
  mode.select_option('paper');assert out.inner_text().startswith('3 nodes:')
  t.fill('9');assert out.inner_text().startswith('4 nodes:')
  w.get_by_role('button',name='Reset trace').click();assert t.input_value()=='7' and not enforce.is_checked() and mode.input_value()=='paper'
  enforce.focus();page.keyboard.press('Space');assert enforce.is_checked();w.get_by_role('button',name='Reset trace').click();states+=6
  from relkit.cvitkovic_l118 import rdb_to_graph,undirected_hops
  edges=[(1,0),(0,2),(3,2),(4,3),(5,1)];times=[None,5,None,None,11,9]
  for cutoff in range(13):
   for method in ['paper','radius']:
    for filtered in [True,False]:
     legal=[(u,v) for u,v in edges if all(not filtered or times[i] is None or times[i]<=cutoff for i in [u,v])]
     expected=rdb_to_graph(6,legal,0)[0] if method=='paper' else undirected_hops(6,legal,0)
     actual=page.evaluate('(x)=>RDBExtractionViz.compute(...x)',[method,cutoff,filtered]);assert actual==expected
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),'Page overflow'
  assert page.locator('figure').count()==3
  page.screenshot(path=f'/tmp/l118-page-{width}.png',full_page=True);w.screenshot(path=f'/tmp/l118-widget-{width}.png');page.locator('figure').nth(1).screenshot(path=f'/tmp/l118-architecture-{width}.png')
 page.set_viewport_size({'width':1000,'height':900});page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('figure img[src^="data:image/png;base64,"]').count()==3
 page.locator('figure').nth(1).screenshot(path='/tmp/l118-notebook-architecture.png')
 nojs=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});plain=nojs.new_page();plain.goto((R/'lessons'/f'{S}.html').as_uri());assert 'NOT_RUN' in plain.inner_text('article') and 'Static trace:' in plain.inner_text('article');assert plain.evaluate('document.documentElement.scrollWidth<=innerWidth+1');plain.emulate_media(media='print');assert plain.locator('figure').count()==3;nojs.close();browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
(P/'_delivery_l118_results.json').write_text('{"status":"RUNNING"}\n')
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0];lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))];checked=0
with tempfile.TemporaryDirectory(prefix='l118-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines);subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/cvitkovic-relational-gnn.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/cvitkovic_l118.py','labs/_run_l118.py','labs/l118-reproduction.md','labs/solutions/'+S+'.ipynb','labs/_pilot_l118_results.json','labs/sources/l118/models/GNN/GCN.py','modal/l118_pilot.py']:assert (stage/rel).exists(),rel
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));base=f'http://127.0.0.1:{server.server_port}'
   page.goto(base+'/index.html');reveal_gallery_link(page, 'a[href="lessons/'+S+'.html"]');assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(base+'/notebooks.html');reveal_gallery_link(page, 'a[href="labs/html/'+S+'.html"]')
   page.goto(base+'/lessons/'+S+'.html');assert '4 nodes' in page.locator('#l118-extraction output').inner_text();browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
r={'status':'PASS','browser_widths':[1200,375],'interactive_states':states,'independent_widget_cases':52,'keyboard_reset':'PASS','print_nojs':'PASS','portable_figures':3,'live_tasks':3,'canonical_definitions':len(canonical),'executed_code_hash':'MATCH','deterministic_rebuild':'EXACT','copied_pages_links':checked,'javascript_errors':errors,'screenshots':'/tmp/l118-*.png','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/'_delivery_l118_results.json').write_text(json.dumps(r,indent=2));print(r)
