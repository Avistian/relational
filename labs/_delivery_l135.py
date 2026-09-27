"""Standalone notebook, deterministic build, real browser and copied Pages checks."""
import ast,base64,functools,hashlib,json,os,re,subprocess,sys,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0135-tuning-on-reg'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert not any(c.outputs for c in student.cells if c.cell_type=='code')
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
digest=hashlib.sha256(code.encode()).hexdigest();assert digest==json.loads((P/'_execution_l135_results.json').read_text())['executed_code_sha256']
gpu=json.loads((P/'_notebook_gpu_l135_results.json').read_text());assert gpu['status']=='PASS' and digest==gpu['code_sha256']
def definitions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
canonical=definitions((P/'relkit/tuning_l135.py').read_text());sol=definitions(code)
for name,node in canonical.items():assert sol[name]==node,name
assert not any('from relkit' in c.source for c in solution.cells if c.cell_type=='code')
images=re.findall('data:image/png;base64,([A-Za-z0-9+/=]+)','\n'.join(c.source for c in student.cells));assert len(images)==5
for data in images:assert base64.b64decode(data).startswith(b'\x89PNG')
paths=[R/'lessons'/f'{S}.html',R/'reference/tuning-on-reg.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb'];before=[sha(p) for p in paths]
subprocess.run([sys.executable,str(P/'_build_l135.py')],check=True,capture_output=True);assert before==[sha(p) for p in paths],'Builder drift'
figs=sorted((P/'figures/l135').glob('*'));before=[sha(p) for p in figs]
subprocess.run([sys.executable,str(P/'_figures_l135.py')],check=True,capture_output=True);assert before==[sha(p) for p in figs],'Figure drift'
# Refresh prepared HTML from the executed notebook after any prose-only update.
html,_=HTMLExporter(template_name='lab').from_notebook_node(solution);(P/'html'/f'{S}.html').write_text(html)
errors=[];states=0;launch=dict(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
with sync_playwright() as pw:
 browser=pw.chromium.launch(**launch);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
 for width in [1200,375]:
  page.set_viewport_size(dict(width=width,height=900));page.goto((R/'lessons'/f'{S}.html').as_uri())
  selection=page.locator('#l135-selection');slider=selection.locator('input')
  for value in [1,2,3,4,5,6]:
   slider.fill(str(value));slider.dispatch_event('input');expected='A' if (2+value)/2<=2.5 else 'B';assert 'Winner: '+expected in selection.inner_text();states+=1
  selection.locator('button').click();assert slider.input_value()=='4'
  slider.focus();page.keyboard.press('ArrowRight');assert slider.input_value()=='4.5';selection.locator('button').click()
  budget=page.locator('#l135-budget');control=budget.locator('input')
  for workers in range(1,31):
   control.fill(str(workers));control.dispatch_event('input');expected='ALLOW' if 6+workers*.203148<=10 else 'REFUSE';assert expected in budget.inner_text();states+=1
  budget.locator('button').click();assert control.input_value()=='12'
  control.focus();page.keyboard.press('ArrowRight');assert control.input_value()=='13';budget.locator('button').click()
  assert page.locator('#warmup button').count()>0
  teach=page.locator('#l135-teachback');assert teach.locator('textarea').count()==1
  teach.locator('textarea').fill('One task and a selected configuration do not cover variation across databases or search decisions.')
  # Screenshot actual layout and every figure at reading width.
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),'Page overflow'
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===5 && xs.every(x=>x.complete&&x.naturalWidth>0)')
  for i in range(5):page.locator('figure').nth(i).screenshot(path=f'/tmp/l135-figure-{i}-{width}.png')
  selection.screenshot(path=f'/tmp/l135-selection-{width}.png');budget.screenshot(path=f'/tmp/l135-budget-{width}.png')
 page.emulate_media(media='print');assert page.locator('article').is_visible();page.pdf(path='/tmp/l135-print.pdf',format='A4',print_background=True);page.emulate_media(media='screen')
 page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('figure img[src^="data:image/png;base64,"]').count()==5
 assert page.locator('figure img').last.evaluate('(x)=>x.getBoundingClientRect().width>=580')
 page.locator('figure').last.screenshot(path='/tmp/l135-notebook-paired.png')
 page.set_viewport_size(dict(width=1000,height=1000));page.locator('figure').last.screenshot(path='/tmp/l135-notebook-paired-desktop.png')
 nojs=browser.new_context(java_script_enabled=False,viewport=dict(width=375,height=900));np=nojs.new_page();np.goto((R/'lessons'/f'{S}.html').as_uri());assert 'B wins' in np.locator('noscript').first.inner_text();assert 'refuse' in np.locator('noscript').last.inner_text();assert not np.evaluate('document.documentElement.scrollWidth>innerWidth+1');nojs.close();browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))];checked=0
with tempfile.TemporaryDirectory(prefix='l135-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/tuning-on-reg.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/tuning_l135.py','labs/relkit/tuning_train_l135.py','labs/l135-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l135/summary.json','labs/_protocol_l135.json','labs/_new_run_l135.py','modal/l135_repro.py']:
  assert (stage/rel).exists(),rel
 assert not (stage/'labs/results/l135').exists()
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(**launch);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);base=f'http://127.0.0.1:{server.server_port}'
   page.goto(base+'/index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]');assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(base+'/notebooks.html');page.wait_for_selector('a[href="labs/html/'+S+'.html"]')
   page.goto(base+'/lessons/'+S+'.html');assert 'Winner: B' in page.locator('#l135-selection-result').inner_text();browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',print_nojs='PASS',portable_figures=5,live_tasks=3,full_training_code_path='PASS; isolated pinned GPU namespace',executed_code_hash=digest,deterministic_rebuild='EXACT',copied_pages_links=checked,javascript_errors=errors,screenshots='/tmp/l135-*.png',live_colab='NOT_CHECKED',deployment='NOT_CHECKED',clean_committed_checkout='NOT_CHECKED; working-tree staging only')
(P/'_delivery_l135_results.json').write_text(json.dumps(r,indent=2));print(r)
