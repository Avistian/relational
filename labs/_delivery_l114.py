"""Canonical notebook alignment, real browser states and exact copied Pages build."""
import ast,functools,hashlib,json,os,re,subprocess,sys,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0114-ogb-error-analysis'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
student=nbformat.read(P/f'{S}.ipynb',as_version=4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert not any(c.outputs for c in student.cells if c.cell_type=='code')
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert 'from relkit' not in code and '__file__' not in code
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_l114_results.json').read_text())['executed_code_sha256']
assert sha(P/'relkit/error_l114.py')==json.loads((P/'_sources_l114.json').read_text())['source_sha256']
def definitions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
canonical={**definitions((P/'relkit/ogb_l112.py').read_text()),**definitions((P/'relkit/error_l114.py').read_text())};sol=definitions(code);stu=definitions('\n\n'.join(c.source for c in student.cells if c.cell_type=='code'))
for name,node in canonical.items():
 assert sol[name]==node,name
 if name not in ['neighborhood_properties','slice_metrics','choose_failure']:assert stu[name]==node,name
paths=[R/'lessons'/f'{S}.html',R/'reference/ogb-error-analysis.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb'];before=[sha(p) for p in paths];subprocess.run([sys.executable,str(P/'_build_l114.py')],check=True,capture_output=True);assert before==[sha(p) for p in paths],'Builder drift'
figs=sorted((P/'figures/l114').glob('*'));before=[sha(p) for p in figs];subprocess.run([sys.executable,str(P/'_figures_l114.py')],check=True,capture_output=True);assert before==[sha(p) for p in figs],'Figure drift'
errors=[];states=0;summary=json.loads((P/'evidence/l114/summary.json').read_text())
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('requestfailed',lambda r:errors.append(r.url))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri());widget=page.locator('[data-error-slices]');pop=widget.get_by_label('Population',exact=True);family=widget.get_by_label('Slice family',exact=True)
  for population in ['valid','test']:
   pop.select_option(population)
   for fam in ['degree','homophily','class','year']:
    family.select_option(fam);rows=[r for r in summary['slice_rows'] if r['population']==population and r['family']==fam];assert widget.locator('table tr').count()==1+len(rows)
    assert ('29,799' if population=='valid' else '48,603') in widget.locator('[aria-live]').inner_text()
    expected=[r for r in rows if r['n']];bars=widget.locator('.error-bar');assert bars.count()==len(expected)*2
    for i,r in enumerate(expected):
     for j,key in enumerate(['gcn','mlp']):assert abs(float(bars.nth(i*2+j).evaluate('(e)=>parseFloat(e.style.width)'))-r[key+'_mean_percent'])<.0001
    states+=1
  widget.get_by_role('button',name='Reset view').click();assert pop.input_value()=='test' and family.input_value()=='homophily'
  family.focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert family.input_value()=='class';widget.get_by_role('button',name='Reset view').click()
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),'Page overflow'
  assert page.locator('figure').count()==4
  page.screenshot(path=f'/tmp/l114-page-{width}.png',full_page=True);widget.screenshot(path=f'/tmp/l114-widget-{width}.png')
  page.locator('figure').first.screenshot(path=f'/tmp/l114-architecture-{width}.png')
 page.set_viewport_size({'width':1100,'height':900});page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('img[src^="data:image/png;base64,"]').count()==4
 for i in range(4):page.locator('img[src^="data:image/png;base64,"]').nth(i).screenshot(path=f'/tmp/l114-notebook-{i}.png')
 nojs=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});plain=nojs.new_page();plain.goto((R/'lessons'/f'{S}.html').as_uri());assert plain.locator('figure').count()==4 and '8,585' in plain.inner_text('article');assert plain.evaluate('document.documentElement.scrollWidth<=innerWidth+1');plain.emulate_media(media='print');assert plain.locator('details p').first.evaluate('(e)=>e.checkVisibility()');plain.screenshot(path='/tmp/l114-print.png',full_page=True);nojs.close();browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
(P/'_delivery_l114_results.json').write_text('{"status":"RUNNING"}\n')
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0];lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))];checked=0
with tempfile.TemporaryDirectory(prefix='l114-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines);subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/ogb-error-analysis.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/error_l114.py','labs/_run_l114.py','labs/l114-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l114/summary.json','labs/evidence/l114/analysis-inputs.npz','labs/sources/l114/mlp.py','modal/l114_repro.py']:assert (stage/rel).exists(),rel
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));base=f'http://127.0.0.1:{server.server_port}'
   page.goto(base+'/index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]');assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(base+'/notebooks.html');page.wait_for_selector('a[href="labs/html/'+S+'.html"]')
   page.goto(base+'/lessons/'+S+'.html');assert '48,603' in page.locator('[data-error-slices] [aria-live]').inner_text();browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
r={'status':'PASS','browser_widths':[1200,375],'widget_states':states,'keyboard_reset':'PASS','print_nojs':'PASS','portable_figures':4,'live_student_tasks':3,'canonical_definitions':len(canonical),'executed_code_hash':'MATCH','deterministic_rebuild':'EXACT','copied_pages_links':checked,'javascript_errors':errors,'screenshots':'/tmp/l114-*.png','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/'_delivery_l114_results.json').write_text(json.dumps(r,indent=2));print(r)
