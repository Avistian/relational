"""Bounded sequential browser checks and copied workflow publication staging."""
from _gallery_delivery import reveal_gallery_link
import ast,functools,hashlib,json,os,re,subprocess,sys,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0134-training-at-scale'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
student=nbformat.read(P/f'{S}.ipynb',as_version=4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert not any(c.outputs for c in student.cells if c.cell_type=='code')
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_l134_results.json').read_text())['executed_code_sha256']
from _profile_provenance_l134 import verify_profile_provenance
verify_profile_provenance(code)
def definitions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
canonical=definitions((P/'relkit/scale_l134.py').read_text());sol=definitions(code);stu=definitions('\n\n'.join(c.source for c in student.cells if c.cell_type=='code'))
for name,node in canonical.items():
 if name in ['reserve_cost','inspect_batch']:continue
 assert sol[name]==node,name
 if name not in ['frontier_bound','audit_queries','profile_summary']:assert stu[name]==node,name
paths=[R/'lessons'/f'{S}.html',R/'reference/training-at-scale.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb'];before=[sha(p) for p in paths]
subprocess.run([sys.executable,str(P/'_build_l134.py')],check=True,capture_output=True);subprocess.run([sys.executable,str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True);assert before==[sha(p) for p in paths],'Builder drift'
figs=sorted((P/'figures/l134').glob('*'));before=[sha(p) for p in figs]
subprocess.run([sys.executable,str(P/'_figures_l134.py')],check=True,capture_output=True);assert before==[sha(p) for p in figs],'Figure drift'
errors=[];states=0
launch=dict(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
with sync_playwright() as pw:
 browser=pw.chromium.launch(**launch);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri())
  w=page.locator('#l134-budget');inputs=w.locator('input');out=w.locator('output')
  for batch in [1,2,8]:
   for f in [0,3,8]:
    for g in [0,2,8]:
     for i,value in enumerate([batch,f,g]):inputs.nth(i).fill(str(value));inputs.nth(i).dispatch_event('input')
     expected=batch+batch*f+2*batch*f*g
     assert f'Total {expected} occurrences;' in out.inner_text();states+=1
  w.get_by_role('button',name='Reset',exact=True).click();assert [inputs.nth(i).input_value() for i in range(3)]==['2','3','2']
  inputs.nth(0).focus();page.keyboard.press('ArrowRight');assert inputs.nth(0).input_value()=='3'
  w.get_by_role('button',name='Reset',exact=True).click()
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),'Page overflow'
  assert page.locator('figure').count()==4
  page.locator('figure img').evaluate_all('(xs)=>xs.forEach(x=>x.loading="eager")');page.wait_for_function('Array.from(document.querySelectorAll("figure img")).every(x=>x.complete && x.naturalWidth>0)');assert page.locator('figure img').evaluate_all('(xs)=>xs.every(x=>x.complete && x.naturalWidth>0)')
  page.locator('figure').nth(1).screenshot(path=f'/tmp/l134-arithmetic-{width}.png');w.screenshot(path=f'/tmp/l134-widget-{width}.png')
 page.emulate_media(media='print');assert page.locator('article').is_visible();assert page.locator('figure').count()==4;page.emulate_media(media='screen')
 page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('figure img[src^="data:image/png;base64,"]').count()==4
 nojs=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});np=nojs.new_page();np.goto((R/'lessons'/f'{S}.html').as_uri());assert '32 occurrences' in np.locator('noscript').inner_text();assert not np.evaluate('document.documentElement.scrollWidth>innerWidth+1');nojs.close();browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))];checked=0
with tempfile.TemporaryDirectory(prefix='l134-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/training-at-scale.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/scale_l134.py','labs/l134-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l134/summary.json','labs/_run_l134.py','labs/evidence/l134/scale/scale.json','modal/l134_repro.py']:
  assert (stage/rel).exists(),rel
 assert not (stage/'labs/results/l134').exists()
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(**launch);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));base=f'http://127.0.0.1:{server.server_port}'
   page.goto(base+'/index.html');reveal_gallery_link(page, 'a[href="lessons/'+S+'.html"]');assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(base+'/notebooks.html');reveal_gallery_link(page, 'a[href="labs/html/'+S+'.html"]')
   page.goto(base+'/lessons/'+S+'.html');assert 'Total 32 occurrences;' in page.locator('#l134-budget output').inner_text();browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',print_nojs='PASS',portable_figures=4,live_tasks=3,full_training_code_path='ARCHIVED_PASS; unchanged training AST, current measurement guards checked locally',canonical_definitions=len(canonical),executed_code_hash='MATCH',deterministic_rebuild='EXACT',copied_pages_links=checked,javascript_errors=errors,screenshots='/tmp/l134-*.png',live_colab='NOT_CHECKED',deployment='NOT_CHECKED',clean_committed_checkout='NOT_CHECKED; working-tree staging only')
(P/'_delivery_l134_results.json').write_text(json.dumps(r,indent=2));print(r)
