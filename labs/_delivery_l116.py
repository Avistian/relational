"""Canonical notebook alignment, real browser states and exact copied Pages build."""
import ast,functools,hashlib,json,os,re,subprocess,sys,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0116-debug-gnn-training'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
student=nbformat.read(P/f'{S}.ipynb',as_version=4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert not any(c.outputs for c in student.cells if c.cell_type=='code')
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert 'from relkit' not in code and '__file__' not in code
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_l116_results.json').read_text())['executed_code_sha256']
assert sha(P/'relkit/debug_l116.py')==json.loads((P/'_sources_l116.json').read_text())['source_sha256']
def definitions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
canonical=definitions((P/'relkit/debug_l116.py').read_text());sol=definitions(code);stu=definitions('\n\n'.join(c.source for c in student.cells if c.cell_type=='code'))
for name,node in canonical.items():
 assert sol[name]==node,name
 if name not in ['training_loss','train_step','seed_loss']:assert stu[name]==node,name
paths=[R/'lessons'/f'{S}.html',R/'reference/debug-gnn-training.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb'];before=[sha(p) for p in paths];subprocess.run([sys.executable,str(P/'_build_l116.py')],check=True,capture_output=True);assert before==[sha(p) for p in paths],'Builder drift'
figs=sorted((P/'figures/l116').glob('*'));before=[sha(p) for p in figs];subprocess.run([sys.executable,str(P/'_figures_l116.py')],check=True,capture_output=True);assert before==[sha(p) for p in figs],'Figure drift'
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('requestfailed',lambda r:errors.append(r.url))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri())
  update=page.locator('[data-gnn-update]');lr=update.locator('input[type=range]');missing=update.locator('input[type=checkbox]');out=update.locator('output')
  assert 'zero parameter movement' in out.inner_text();assert update.locator('tbody tr').last.locator('td').nth(3).inner_text()=='1.0000'
  missing.uncheck();assert update.locator('tbody tr').last.locator('td').nth(3).inner_text()=='0.4096';states+=2
  lr.fill('0');assert 'Both paths are frozen' in out.inner_text();states+=1
  update.get_by_role('button',name='Reset update').click();assert lr.input_value()=='0.2' and missing.is_checked()
  missing.focus();page.keyboard.press('Space');assert not missing.is_checked();update.get_by_role('button',name='Reset update').click()
  leak=page.locator('[data-gnn-leak]');sel=leak.locator('select');flip=leak.locator('input');out=leak.locator('output')
  flip.check();assert 'change 0.000000' in out.inner_text();sel.select_option('all');assert 'change -0.313191' in out.inner_text();states+=2
  leak.get_by_role('button',name='Reset label probe').click();assert sel.input_value()=='train' and not flip.is_checked()
  smooth=page.locator('#l116-smoothing');slider=smooth.locator('input[type=range]');slider.fill('64');assert 'Depth 64' in smooth.inner_text();smooth.locator('select').select_option('mean');smooth.locator('input[type=checkbox]').check();assert 'isolated component' in smooth.inner_text();smooth.get_by_role('button',name='Reset',exact=True).click();assert slider.input_value()=='1';states+=3
  # Independent Python arithmetic for every branch, including zero learning rate.
  for lr0 in [0,.2,1,-1,2]:
   for omit in [False,True]:
    actual=page.evaluate('(a)=>GNNDiagnostics.update(...a)',[lr0,omit]);theta=1.;rate=max(0,min(1,lr0))
    for row in actual:
     before=theta
     if not omit:theta-=rate*theta
     assert abs(row['theta']-theta)<1e-12 and abs(row['delta']-(theta-before))<1e-12
  import math
  for scope in ['train','all']:
   for perturb in [False,True]:
    actual=page.evaluate('(a)=>GNNDiagnostics.leakage(...a)',[scope,perturb]);probs=[[.9,.1],[.2,.8],[.7,.3],[.4,.6]];labels=[0,1,0,1] if perturb else [0,1,1,0];ids=[0,1] if scope=='train' else [0,1,2,3];expected=-sum(math.log(probs[i][labels[i]]) for i in ids)/len(ids);assert abs(actual['loss']-expected)<1e-12
  for mode in ['symmetric','mean']:
   for cut in [False,True]:
    for depth in [0,1,8,64]:
     aa=[[1,1,0],[1,1,0 if cut else 1],[0,0 if cut else 1,1]];dd=list(map(sum,aa));hh=[2.,4.,8.]
     for _ in range(depth):hh=[sum(aa[i][j]*hh[j]/(dd[i] if mode=='mean' else math.sqrt(dd[i]*dd[j])) for j in range(3)) for i in range(3)]
     actual=page.evaluate('(a)=>OversmoothingViz.compute(...a).h',[depth,mode,cut]);assert max(abs(a-b) for a,b in zip(actual,hh))<1e-10
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),'Page overflow'
  assert page.locator('figure').count()==6
  page.screenshot(path=f'/tmp/l116-page-{width}.png',full_page=True);update.screenshot(path=f'/tmp/l116-update-{width}.png');leak.screenshot(path=f'/tmp/l116-leak-{width}.png');smooth.screenshot(path=f'/tmp/l116-smoothing-{width}.png');page.locator('figure').first.screenshot(path=f'/tmp/l116-trace-{width}.png')
 page.set_viewport_size({'width':1100,'height':900});page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('figure img[src^="data:image/png;base64,"]').count()==6
 for i in range(6):page.locator('figure img[src^="data:image/png;base64,"]').nth(i).screenshot(path=f'/tmp/l116-notebook-{i}.png')
 nojs=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});plain=nojs.new_page();plain.goto((R/'lessons'/f'{S}.html').as_uri());assert plain.locator('figure').count()==6 and '72.0200%' in plain.inner_text('article');assert plain.evaluate('document.documentElement.scrollWidth<=innerWidth+1');plain.emulate_media(media='print');assert plain.locator('details p').first.evaluate('(e)=>e.checkVisibility()');plain.screenshot(path='/tmp/l116-print.png',full_page=True);nojs.close();browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
(P/'_delivery_l116_results.json').write_text('{"status":"RUNNING"}\n')
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0];lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))];checked=0
with tempfile.TemporaryDirectory(prefix='l116-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines);subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/debug-gnn-training.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/debug_l116.py','labs/_run_l116.py','labs/l116-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l116/summary.json','labs/sources/l116/gnn.py','modal/l116_repro.py']:assert (stage/rel).exists(),rel
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));base=f'http://127.0.0.1:{server.server_port}'
   page.goto(base+'/index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]');assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(base+'/notebooks.html');page.wait_for_selector('a[href="labs/html/'+S+'.html"]')
   page.goto(base+'/lessons/'+S+'.html');assert 'zero parameter movement' in page.locator('[data-gnn-update] output').inner_text();browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
r={'status':'PASS','browser_widths':[1200,375],'widget_states':states,'keyboard_reset':'PASS','print_nojs':'PASS','portable_figures':6,'live_student_tasks':3,'canonical_definitions':len(canonical),'executed_code_hash':'MATCH','deterministic_rebuild':'EXACT','copied_pages_links':checked,'javascript_errors':errors,'screenshots':'/tmp/l116-*.png','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/'_delivery_l116_results.json').write_text(json.dumps(r,indent=2));print(r)
