"""Actual notebook, browser and copied Pages validation; no live frontend claims."""
from _gallery_delivery import reveal_gallery_link
import ast,base64,functools,hashlib,json,os,re,subprocess,sys,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0136-leaderboard-literacy'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert not any(c.outputs for c in student.cells if c.cell_type=='code')
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');digest=hashlib.sha256(code.encode()).hexdigest()
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert digest==json.loads((P/'_execution_l136_results.json').read_text())['executed_code_sha256']
for name in ['_notebook_gpu_l136_results.json','_notebook_replay_l136_results.json']:
 r=json.loads((P/name).read_text());assert r['status']=='PASS' and r['code_sha256']==digest

def definitions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
canonical=definitions((P/'relkit/leaderboard_l136.py').read_text());sol=definitions(code)
for name,node in canonical.items():assert sol[name]==node,name
assert 'from relkit' not in code
images=re.findall('data:image/png;base64,([A-Za-z0-9+/=]+)','\n'.join(c.source for c in student.cells));assert len(images)==6
for data in images:assert base64.b64decode(data).startswith(b'\x89PNG')
paths=[R/'lessons'/f'{S}.html',R/'reference/leaderboard-literacy.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb'];before=[sha(p) for p in paths]
subprocess.run([sys.executable,str(P/'_build_l136.py')],check=True,capture_output=True);subprocess.run([sys.executable,str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True);assert before==[sha(p) for p in paths],'Builder drift'
figs=sorted((P/'figures/l136').glob('*'));before=[sha(p) for p in figs]
subprocess.run([sys.executable,str(P/'_figures_l136.py')],check=True,capture_output=True);assert before==[sha(p) for p in figs],'Figure drift'
html,_=HTMLExporter(template_name='lab').from_notebook_node(solution);(P/'html'/f'{S}.html').write_text(html)
errors=[];states=0;launch=dict(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
with sync_playwright() as pw:
 browser=pw.chromium.launch(**launch);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
 for width in [1200,375]:
  page.set_viewport_size(dict(width=width,height=900));page.goto((R/'lessons'/f'{S}.html').as_uri())
  scale=page.locator('#l136-scale');slider=scale.locator('input')
  for value in [1,1.5,2,2.5,3,3.5,4]:
   slider.fill(str(value));slider.dispatch_event('input');assert f'{1/value:.3f}' in scale.locator('.audit-result').inner_text();states+=1
  scale.locator('button').click();assert slider.input_value()=='2';slider.focus();page.keyboard.press('ArrowRight');assert slider.input_value()=='2.5';scale.locator('button').click()
  coverage=page.locator('#l136-coverage');control=coverage.locator('input')
  for n in [1,2]:
   control.fill(str(n));control.dispatch_event('input');assert ('COMPLETE' if n==2 else 'REJECT') in coverage.locator('.audit-result').inner_text();states+=1
  coverage.locator('button').click();assert control.input_value()=='2';control.focus();page.keyboard.press('ArrowLeft');assert control.input_value()=='1';coverage.locator('button').click()
  assert page.locator('#warmup button').count()==0
  teach=page.locator('#l136-teachback');assert teach.locator('textarea').count()==1
  teach.locator('textarea').fill('Exact scores establish evaluation; training inputs, temporal regimes and search budgets may still differ.')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),'Page overflow'
  page.locator('figure img').evaluate_all('(xs)=>xs.forEach(x=>x.loading="eager")');page.wait_for_function('Array.from(document.querySelectorAll("figure img")).every(x=>x.complete && x.naturalWidth>0)');assert page.locator('figure img').evaluate_all('(xs)=>xs.length===6 && xs.every(x=>x.complete&&x.naturalWidth>0)')
  page.evaluate('window.scrollTo(0,0)');page.screenshot(path=f'/tmp/l136-top-{width}.png')
  for i in range(6):
   figure=page.locator('figure').nth(i);figure.screenshot(path=f'/tmp/l136-figure-{i}-{width}.png')
   if width==375:
    assert figure.evaluate('(x)=>x.scrollWidth>x.clientWidth')
    figure.evaluate('(x)=>x.scrollLeft=x.scrollWidth');assert figure.evaluate('(x)=>x.scrollLeft>0')
    figure.screenshot(path=f'/tmp/l136-figure-{i}-mobile-right.png');figure.evaluate('(x)=>x.scrollLeft=0')
  scale.screenshot(path=f'/tmp/l136-scale-{width}.png');coverage.screenshot(path=f'/tmp/l136-coverage-{width}.png')
 page.emulate_media(media='print');assert page.locator('article').is_visible();page.pdf(path='/tmp/l136-print.pdf',format='A4',print_background=True);page.emulate_media(media='screen')
 page.set_viewport_size(dict(width=1000,height=1000));page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('figure img[src^="data:image/png;base64,"]').count()==6
 for i in [0,3,5]:page.locator('figure').nth(i).screenshot(path=f'/tmp/l136-notebook-{i}.png')
 nojs=browser.new_context(java_script_enabled=False,viewport=dict(width=375,height=900));np=nojs.new_page();np.goto((R/'lessons'/f'{S}.html').as_uri());assert '0.500' in np.locator('noscript').first.inner_text();assert 'REJECT' in np.locator('noscript').last.inner_text();assert not np.evaluate('document.documentElement.scrollWidth>innerWidth+1');nojs.close();browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))];checked=0
with tempfile.TemporaryDirectory(prefix='l136-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/leaderboard-literacy.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/leaderboard_l136.py','labs/l136-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l136/leaderboard.json','labs/evidence/l136/training.json','labs/_new_run_l136.py','modal/l136_repro.py','labs/sources/l136/kapso.zip']:
  assert (stage/rel).exists(),rel
 assert not (stage/'labs/results/l136').exists()
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(**launch);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));base=f'http://127.0.0.1:{server.server_port}'
   page.goto(base+'/index.html');reveal_gallery_link(page, 'a[href="lessons/'+S+'.html"]');assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(base+'/notebooks.html');reveal_gallery_link(page, 'a[href="labs/html/'+S+'.html"]')
   page.goto(base+'/lessons/'+S+'.html');assert '0.500' in page.locator('#l136-scale .audit-result').inner_text();browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',print_nojs='PASS',portable_figures=6,live_tasks=3,full_training_code_path='PASS; five isolated GPU fits',full_evaluation_code_path='PASS; all27 task-entry pairs in isolated directory',executed_code_hash=digest,deterministic_rebuild='EXACT',copied_pages_links=checked,javascript_errors=errors,screenshots='/tmp/l136-*.png',live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l136_results.json').write_text(json.dumps(r,indent=2));print(r)
