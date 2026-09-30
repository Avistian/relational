"""Browser controls, standalone notebooks, deterministic generation and copied Pages."""
import ast,hashlib,json,os,re,subprocess,sys,tempfile,threading,functools
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from _gallery_delivery import reveal_gallery_link
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0142-many-to-many-edge-pathology'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert all(not c.outputs for c in student.cells if c.cell_type=='code')
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');digest=hashlib.sha256(code.encode()).hexdigest()
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert digest==json.loads((P/'_execution_l142_results.json').read_text())['executed_code_sha256']
assert 'from relkit' not in code and 'from _full_l142' not in code
# Every canonical class/live function and trainer body remains visible.
inline={n.name:n for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for file in ['relkit/pathology_l142.py','relkit/relgnn_l142.py','_full_l142.py']:
 for n in ast.parse((P/file).read_text()).body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)):
   # Remove only the harness import relocated into a visible preceding cell.
   if n.name=='full_run':n.body=[x for x in n.body if not isinstance(x,ast.ImportFrom)]
   assert ast.dump(n,include_attributes=False)==ast.dump(inline[n.name],include_attributes=False),(file,n.name)
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
 page.on('requestfailed',lambda r:errors.append(r.url))
 for width in [1200,375]:
  page.set_viewport_size(dict(width=width,height=900));page.goto((R/'lessons'/f'{S}.html').as_uri())
  for id,hub in [('l142-walks',False),('l142-hub',True)]:
   host=page.locator('#'+id)
   for source,fact,destination,noise in [(2,1,3,8),(0,0,1,0),(1,0,0,0),(0,1,0,0),(8,1,3,2)]:
    for key,value in [('s',source),('f',fact),('d',destination)]+([('n',noise)] if hub else []):
     inp=host.locator('[data-key="'+key+'"]');inp.fill(str(value));inp.dispatch_event('input')
    expected=source+2*fact+2*destination+(noise if hub else 0)
    text=host.locator('.path-readout').inner_text()
    assert f'Ordinary after layer 2: {expected}.' in text
    assert f'Source-route composite: {source+fact+destination}.' in text;states+=1
   host.locator('[data-reset]').click();assert host.locator('[data-key="s"]').input_value()=='2'
   inp=host.locator('[data-key="s"]');inp.focus();page.keyboard.press('ArrowRight');assert inp.input_value()=='3'
   host.locator('[data-reset]').click()
   if hub:
    host.locator('[data-swap]').click();assert 'Ordinary after layer 2: 18.' in host.inner_text() and 'Source-route composite: 12.' in host.inner_text();states+=1
    host.locator('[data-reset]').click()
  assert page.locator('#warmup button').count()>0 and page.locator('#l142-teachback textarea').count()==1
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===4&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
  page.evaluate('window.scrollTo(0,0)');page.screenshot(path=f'/tmp/l142-top-{width}.png')
  for i in range(4):page.locator('figure').nth(i).screenshot(path=f'/tmp/l142-figure-{i}-{width}.png')
  page.locator('#l142-hub').screenshot(path=f'/tmp/l142-hub-{width}.png')
 page.emulate_media(media='print');page.set_viewport_size(dict(width=1100,height=900));page.screenshot(path='/tmp/l142-print.png',full_page=True);page.emulate_media(media='screen')
 page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('figure img[src^="data:image/png;base64,"]').count()==4
 page.locator('figure').nth(2).screenshot(path='/tmp/l142-notebook-architecture.png')
 nojs=browser.new_context(java_script_enabled=False,viewport=dict(width=375,height=900));pg=nojs.new_page();pg.goto((R/'lessons'/f'{S}.html').as_uri());assert 'Baseline s=2' in pg.locator('noscript').first.inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');nojs.close();browser.close()
assert not errors,errors
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0];lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))];checked=0
with tempfile.TemporaryDirectory(prefix='l142-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/many-to-many-edge-pathology.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/relgnn_l142.py','labs/l142-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l142/training.json','labs/_full_l142.py','modal/l142_repro.py','labs/sources/l141/examples__relgnn_model.py']:
  assert (stage/rel).is_file()
# Manifest discovery is checked over HTTP, where gallery fetches work.
class QuietHandler(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(QuietHandler,directory=str(R)))
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
  page.goto(f'http://127.0.0.1:{server.server_port}/index.html')
  link=reveal_gallery_link(page,'a[href="lessons/'+S+'.html"]');assert link.is_visible()
  page.goto(f'http://127.0.0.1:{server.server_port}/notebooks.html')
  link=reveal_gallery_link(page,'a[href="labs/html/'+S+'.html"]');assert link.is_visible();browser.close()
finally:server.shutdown();server.server_close()
paths=[R/'lessons'/f'{S}.html',R/'reference/many-to-many-edge-pathology.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb']+list((P/'figures/l142').glob('*'))
before={str(p):sha(p) for p in paths}
subprocess.run([sys.executable,str(P/'_figures_l142.py')],cwd=R,check=True,capture_output=True)
subprocess.run([sys.executable,str(P/'_build_l142.py')],cwd=R,check=True,capture_output=True)
assert before=={str(p):sha(p) for p in paths},'Nondeterministic build'
report=dict(status='PASS',browser_states=states,widths=[1200,375],keyboard_reset=True,no_javascript=True,print=True,portable_figures=4,copied_pages_links=checked,deterministic_build=True,console_errors=errors,manifest_galleries=True,canonical_inline_ast='PASS',notebook_code_sha256=digest,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l142_results.json').write_text(json.dumps(report,indent=2));print(report)
