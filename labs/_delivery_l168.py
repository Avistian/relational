"""Browser, source parity, portable notebook, copied Pages and deterministic checks."""
import signal
signal.alarm(600)
import ast,functools,hashlib,json,os,re,subprocess,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
from relkit.generalization_l168 import transfer_regime
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0168-cross-database-generalization'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert all(not c.outputs for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert sum(c.source.count('data:image/png;base64,') for c in solution.cells)==3
assert not any(any('def '+name+'(' in c.source for name in ['transfer_regime','paired_gains','database_macro']) for c in student.cells if c.cell_type=='code' and 'raise NotImplementedError' not in c.source)
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_l168_results.json').read_text())['executed_code_sha256']
function_cells={ast.dump(n,include_attributes=False) for c in solution.cells if c.cell_type=='code' for n in ast.parse(c.source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for file in ['relkit/generalization_l168.py','_audit_l168.py','_check_l168.py','_run_l168.py','_fetch_l168.py']:
 for node in ast.parse((P/file).read_text()).body:
  if isinstance(node,(ast.FunctionDef,ast.ClassDef)):assert ast.dump(node,include_attributes=False) in function_cells,(file,node.name)
for source in json.loads((P/'sources/l168/source-ledger.json').read_text())['sources']:
 assert hashlib.sha256((P/'sources/l168'/(source['name']+'.html')).read_bytes()).hexdigest()==source['sha256']
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('requestfailed',lambda req:errors.append(req.url))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/(S+'.html')).as_uri());host=page.locator('#cross-explorer')
  for exposure in ['unknown','excluded','seen']:
   for labels in [0,512]:
    for updates in [0,5]:
     host.locator('[data-exposure]').select_option(exposure);host.locator('[data-labels]').select_option(str(labels));host.locator('[data-updates]').select_option(str(updates))
     if updates and not labels:
      assert host.get_attribute('data-adaptation')=='INVALID_SUPERVISED_PROTOCOL'
     else:
      expected=transfer_regime(['trial'] if exposure=='seen' else [],'trial',exposure=='excluded',labels,updates)
      assert host.get_attribute('data-adaptation')==expected['adaptation'];assert host.get_attribute('data-holdout')==expected['database_holdout']
     assert 'Fixed baseline' in host.inner_text();states+=1
  host.locator('[data-reset]').click();assert host.get_attribute('data-adaptation')=='FEW_SHOT_ICL';assert host.get_attribute('data-holdout')=='NOT_ESTABLISHED'
  control=host.locator('[data-exposure]');control.focus();page.keyboard.press('ArrowDown');page.keyboard.press('Tab');assert control.input_value()=='excluded'
  host.locator('[data-reset]').click();assert control.input_value()=='unknown'
  assert page.locator('#warmup button').count()>0
  predict=page.locator('#predict');assert predict.locator('.predict-reveal').is_disabled()
  labels=[x.inner_text() for x in predict.locator('.predict-option').all()]
  predict.locator('[data-value="few"]').click();predict.locator('.predict-reveal').click();assert '512' in predict.locator('.predict-outcome').inner_text()
  assert page.locator('#teachback textarea').count()==1
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===3&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
  page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l168-top-{width}.png');host.screenshot(path=f'/tmp/l168-intervention-{width}.png')
  for i in range(3):page.locator('figure').nth(i).screenshot(path=f'/tmp/l168-figure-{i}-{width}.png')
 page.emulate_media(media='print');assert page.locator('.cross-controls').first.evaluate('(x)=>getComputedStyle(x).display')=='none'
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert pg.locator('noscript').count()==1;assert '24,750' in pg.locator('body').inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.set_viewport_size({'width':1000,'height':900});page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==3
 assert 'Authenticated both databases' in page.locator('body').inner_text()
 page.locator('img[src^="data:image/png"]').first.screenshot(path='/tmp/l168-notebook-figure.png');browser.close()
assert not errors,errors
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start()
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 for path,selector in [('index.html','a[href="lessons/'+S+'.html"]'),('notebooks.html','a[href="labs/html/'+S+'.html"]')]:
  page.goto('http://127.0.0.1:'+str(server.server_port)+'/'+path);reveal_gallery_link(page,selector)
 browser.close()
server.shutdown();server.server_close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[l[10:] for l in block.splitlines() if l.startswith('          ')];lines=[l for l in lines if not l.startswith(('VER=','sed -i'))];count=0
with tempfile.TemporaryDirectory(prefix='l168-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),l) for l in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/(S+'.html'),stage/'reference/cross-database-generalization.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   dest=(path.parent/unquote(part.path)).resolve();assert dest.exists(),str(dest);count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/cross-database-generalization.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'evidence/l168/report.json',P/'evidence/l168/report.md']+sorted((P/'figures/l168').iterdir())
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l168.py')],check=True,capture_output=True)
after=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
assert before==after,'Builder changed: '+str([str(p) for p,a,b in zip(paths,before,after) if a!=b])
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',no_js='PASS',print='PASS',
       inline_source_parity='PASS',primary_reading_snapshots_checked=3,portable_figures=3,notebook_code_cells=sum(c.cell_type=='code' for c in solution.cells),
       copied_pages_links=count,deterministic_build='PASS',manifest_galleries='PASS',historical_fidelity='NOT_ESTABLISHED',javascript_errors=errors,
       live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l168_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
