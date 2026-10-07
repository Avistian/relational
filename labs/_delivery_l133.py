"""Bounded sequential browser checks and copied workflow publication staging."""
from _gallery_delivery import reveal_gallery_link
import ast,functools,hashlib,json,os,re,subprocess,sys,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0133-hetero-conv-reg'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
student=nbformat.read(P/f'{S}.ipynb',as_version=4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert not any(c.outputs for c in student.cells if c.cell_type=='code')
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_l133_results.json').read_text())['executed_code_sha256']
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_notebook_gpu_l133_results.json').read_text())['code_sha256']
def definitions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
canonical=definitions((P/'relkit/hetero_l133.py').read_text());sol=definitions(code);stu=definitions('\n\n'.join(c.source for c in student.cells if c.cell_type=='code'))
for name,node in canonical.items():
 assert sol[name]==node,name
 if name not in ['sum_neighbors','relation_output','merge_relations']:assert stu[name]==node,name
paths=[R/'lessons'/f'{S}.html',R/'reference/hetero-conv-reg.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb'];before=[sha(p) for p in paths]
subprocess.run([sys.executable,str(P/'_build_l133.py')],check=True,capture_output=True);subprocess.run([sys.executable,str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True);assert before==[sha(p) for p in paths],'Builder drift'
figs=sorted((P/'figures/l133').glob('*'));before=[sha(p) for p in figs]
subprocess.run([sys.executable,str(P/'_figures_l133.py')],check=True,capture_output=True);assert before==[sha(p) for p in figs],'Figure drift'
errors=[];states=0
launch=dict(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
with sync_playwright() as pw:
 browser=pw.chromium.launch(**launch);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri())
  w=page.locator('#l133-layer');select=w.locator('select');slider=w.locator('input');out=w.locator('output')
  for mode in ['present','empty','absent']:
   select.select_option(mode)
   for root in range(9):
    slider.fill(str(root));slider.dispatch_event('input');expected=7+3*root+(0 if mode=='absent' else (2.5 if mode=='present' else 0)-1-2*root)
    assert f'total {expected:g}.' in out.inner_text();states+=1
  w.get_by_role('button',name='Reset',exact=True).click();assert select.input_value()=='present' and slider.input_value()=='4'
  slider.focus();page.keyboard.press('ArrowRight');assert slider.input_value()=='5'
  w.get_by_role('button',name='Reset',exact=True).click()
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),'Page overflow'
  assert page.locator('figure').count()==5
  page.locator('figure img').evaluate_all('(xs)=>xs.forEach(x=>x.loading="eager")');page.wait_for_function('Array.from(document.querySelectorAll("figure img")).every(x=>x.complete && x.naturalWidth>0)');assert page.locator('figure img').evaluate_all('(xs)=>xs.every(x=>x.complete && x.naturalWidth>0)')
  page.locator('figure').nth(1).screenshot(path=f'/tmp/l133-arithmetic-{width}.png');w.screenshot(path=f'/tmp/l133-widget-{width}.png')
 page.emulate_media(media='print');assert page.locator('article').is_visible();assert page.locator('figure').count()==5;page.emulate_media(media='screen')
 page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('figure img[src^="data:image/png;base64,"]').count()==4
 nojs=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});np=nojs.new_page();np.goto((R/'lessons'/f'{S}.html').as_uri());assert 'empty →10' in np.locator('noscript').inner_text();assert not np.evaluate('document.documentElement.scrollWidth>innerWidth+1');nojs.close();browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))];checked=0
with tempfile.TemporaryDirectory(prefix='l133-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/hetero-conv-reg.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/hetero_l133.py','labs/l133-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l133/summary.json','labs/_run_l133.py','labs/sources/l133/failed-pilot/failure.json','modal/l133_repro.py']:
  assert (stage/rel).exists(),rel
 assert not (stage/'labs/results/l133').exists()
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(**launch);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));base=f'http://127.0.0.1:{server.server_port}'
   page.goto(base+'/index.html');reveal_gallery_link(page, 'a[href="lessons/'+S+'.html"]');assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(base+'/notebooks.html');reveal_gallery_link(page, 'a[href="labs/html/'+S+'.html"]')
   page.goto(base+'/lessons/'+S+'.html');assert 'total 12.5.' in page.locator('#l133-layer output').inner_text();browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',print_nojs='PASS',portable_figures=4,live_tasks=3,full_training_code_path='PASS; separate pinned GPU namespace',canonical_definitions=len(canonical),executed_code_hash='MATCH',deterministic_rebuild='EXACT',copied_pages_links=checked,javascript_errors=errors,screenshots='/tmp/l133-*.png',live_colab='NOT_CHECKED',deployment='NOT_CHECKED',clean_committed_checkout='NOT_CHECKED; working-tree staging only')
(P/'_delivery_l133_results.json').write_text(json.dumps(r,indent=2));print(r)
