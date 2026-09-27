"""Notebook identity, responsive interactions, print/no-JS and copied Pages checks."""
import ast,base64,functools,hashlib,json,os,re,subprocess,sys,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0137-error-analysis-reg'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert not any(c.outputs for c in student.cells if c.cell_type=='code')
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');digest=hashlib.sha256(code.encode()).hexdigest()
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert digest==json.loads((P/'_execution_l137_results.json').read_text())['executed_code_sha256']
for name in ['_notebook_gpu_l137_results.json','_notebook_fe_l137_results.json']:
 r=json.loads((P/name).read_text());assert r['status']=='PASS' and r['code_sha256']==digest
canonical={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((P/'relkit/error_reg_l137.py').read_text()).body if isinstance(n,ast.FunctionDef)}
sol={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
for name,node in canonical.items():assert sol[name]==node,name
assert 'from relkit' not in code
images=re.findall('data:image/png;base64,([A-Za-z0-9+/=]+)','\n'.join(c.source for c in student.cells));assert len(images)==5
for data in images:assert base64.b64decode(data).startswith(b'\x89PNG')
paths=[R/'lessons'/f'{S}.html',R/'reference/error-analysis-reg.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb'];before=[sha(p) for p in paths]
subprocess.run([sys.executable,str(P/'_build_l137.py')],check=True,capture_output=True);assert before==[sha(p) for p in paths],'Builder drift'
figs=sorted((P/'figures/l137').glob('*'));before=[sha(p) for p in figs]
subprocess.run([sys.executable,str(P/'_figures_l137.py')],check=True,capture_output=True);assert before==[sha(p) for p in figs],'Figure drift'
html,_=HTMLExporter(template_name='lab').from_notebook_node(solution);(P/'html'/f'{S}.html').write_text(html)
errors=[];states=0;launch=dict(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
with sync_playwright() as pw:
 browser=pw.chromium.launch(**launch);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
 for width in [1200,375]:
  page.set_viewport_size(dict(width=width,height=900));page.goto((R/'lessons'/f'{S}.html').as_uri())
  pair=page.locator('#l137-pair');slider=pair.locator('input')
  for value in range(6,15):
   slider.fill(str(value));slider.dispatch_event('input');assert f'{(4-abs(value-10))/2:.2f}' in pair.locator('.audit-result').inner_text();states+=1
  pair.locator('button').click();assert slider.input_value()=='14';slider.focus();page.keyboard.press('ArrowLeft');assert slider.input_value()=='13';pair.locator('button').click()
  cluster=page.locator('#l137-cluster');control=cluster.locator('select')
  for value,expected in [('AA','0 / 4 queries = 0'),('AB','6 / 3 queries = 2'),('BB','12 / 2 queries = 6')]:
   control.select_option(value);assert expected in cluster.locator('.audit-result').inner_text();states+=1
  cluster.locator('button').click();assert control.input_value()=='AB';control.focus();page.keyboard.press('ArrowDown');assert control.input_value()=='BB';cluster.locator('button').click()
  assert page.locator('#warmup button').count()>0
  teach=page.locator('#l137-teachback');teach.locator('textarea').fill('A validation-selected association does not isolate architecture, optimization or feature access; test support and a controlled intervention are needed.')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),'Page overflow'
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===5 && xs.every(x=>x.complete&&x.naturalWidth>0)')
  page.evaluate('window.scrollTo(0,0)');page.screenshot(path=f'/tmp/l137-top-{width}.png')
  for i in range(5):
   figure=page.locator('figure').nth(i);figure.screenshot(path=f'/tmp/l137-figure-{i}-{width}.png')
   if width==375:
    assert figure.evaluate('(x)=>x.scrollWidth>x.clientWidth');figure.evaluate('(x)=>x.scrollLeft=x.scrollWidth');assert figure.evaluate('(x)=>x.scrollLeft>0');figure.screenshot(path=f'/tmp/l137-figure-{i}-mobile-right.png');figure.evaluate('(x)=>x.scrollLeft=0')
  pair.screenshot(path=f'/tmp/l137-pair-{width}.png');cluster.screenshot(path=f'/tmp/l137-cluster-{width}.png')
 page.emulate_media(media='print');page.pdf(path='/tmp/l137-print.pdf',format='A4',print_background=True);page.emulate_media(media='screen')
 page.set_viewport_size(dict(width=1000,height=1000));page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('figure img[src^="data:image/png;base64,"]').count()==5
 for i in [0,2,4]:page.locator('figure').nth(i).screenshot(path=f'/tmp/l137-notebook-{i}.png')
 nojs=browser.new_context(java_script_enabled=False,viewport=dict(width=375,height=900));np=nojs.new_page();np.goto((R/'lessons'/f'{S}.html').as_uri());assert 'mean gap0' in np.locator('noscript').first.inner_text();assert '6/3=2' in np.locator('noscript').last.inner_text();assert not np.evaluate('document.documentElement.scrollWidth>innerWidth+1');nojs.close();browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))];checked=0
with tempfile.TemporaryDirectory(prefix='l137-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/error-analysis-reg.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/error_reg_l137.py','labs/l137-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l137/errors.json','labs/evidence/l137/training.json','labs/_new_run_l137.py','modal/l137_repro.py']:
  assert (stage/rel).exists(),rel
 assert not (stage/'labs/results/l137').exists()
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(**launch);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));base=f'http://127.0.0.1:{server.server_port}'
   page.goto(base+'/index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]');assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(base+'/notebooks.html');page.wait_for_selector('a[href="labs/html/'+S+'.html"]')
   page.goto(base+'/lessons/'+S+'.html');assert '0.00' in page.locator('#l137-pair .audit-result').inner_text();browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',print_nojs='PASS',portable_figures=5,live_tasks=3,full_training_code_paths='PASS:five GNN fits and50FE trials in isolated pinned runtimes',executed_code_hash=digest,deterministic_rebuild='EXACT',copied_pages_links=checked,javascript_errors=errors,screenshots='/tmp/l137-*.png',live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l137_results.json').write_text(json.dumps(r,indent=2));print(r)
