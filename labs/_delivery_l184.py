"""Real browser coverage, inline-source parity and deterministic output checks."""
import ast,functools,hashlib,itertools,json,os,subprocess,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
from relkit.gelgt_l184 import attention_trace
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0184-gelgt-temporal-attention'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
solcode='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');stucode='\n\n'.join(c.source for c in student.cells if c.cell_type=='code')
assert stucode.count('NotImplementedError')==3 and 'NotImplementedError' not in solcode
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
defs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(solcode).body if isinstance(n,ast.FunctionDef)}
for file,names in [('relkit/gelgt_l184.py',None),('_check_l184.py',None),('_audit_l184.py',['audit184']),('_verify_l184.py',['independent184'])]:
 for n in ast.parse((P/file).read_text()).body:
  if isinstance(n,ast.FunctionDef) and (names is None or n.name in names):assert defs[n.name]==ast.dump(n,include_attributes=False),(file,n.name)
markdown='\n'.join(c.source for c in solution.cells if c.cell_type=='markdown')
for file in ['model.py','local_module.py','encoders.py','codebook.py','utils.py','main_node_ddp.py']:assert (P/'sources/l184/upstream'/file).read_text() in markdown
execution=json.loads((P/'_execution_l184_results.json').read_text());assert execution['executed_code_sha256']==hashlib.sha256(solcode.encode()).hexdigest()
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':1000});page.goto((R/'lessons'/(S+'.html')).as_uri());host=page.locator('#gelgt-viz')
  assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+2'), 'page overflow'
  assert host.locator('.gelgt-scroll').evaluate('(e)=>e.scrollWidth<=e.clientWidth+2'), 'attention columns clipped'
  for cutoff,center,sigma,proj in itertools.product([9,10,20],[0,2,4],[1,2,4],[-1,0,1]):
   for k,v in dict(cutoff=cutoff,center=center,width=sigma,projection=proj).items():host.locator('[data-control="'+k+'"]').select_option(str(v))
   expected=attention_trace([0,2,4],center,sigma,proj)
   actual=page.evaluate('([c,w,p])=>GelGTViz.trace(c,w,p)',[center,sigma,proj])
   assert abs(actual['output']-expected['output'])<1e-12
   assert ('day 15 admitted' if cutoff==20 else 'day 15 excluded') in host.locator('[data-view="access"]').inner_text()
   assert f"{expected['output']:.3f}" in host.locator('[data-view="output"]').inner_text();states+=1
  host.locator('[data-reset]').click();assert host.locator('[data-control="center"]').input_value()=='2'
  host.locator('[data-control="center"]').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Tab');assert host.locator('[data-control="center"]').input_value()=='4'
  host.locator('[data-reset]').click();host.screenshot(path=f'/tmp/l184-widget-{width}.png');page.screenshot(path=f'/tmp/l184-lesson-{width}.png')
  for img in page.locator('article figure img').all():assert img.evaluate('(i)=>i.complete && i.naturalWidth>0')
 page.emulate_media(media='print');assert '8,712' in page.locator('body').inner_text()
 page.emulate_media(media='screen');page.set_viewport_size({'width':1050,'height':900});page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==3
 page.locator('img[src^="data:image/png"]').first.screenshot(path='/tmp/l184-notebook-architecture.png')
 context=browser.new_context(java_script_enabled=False);np=context.new_page();np.goto((R/'lessons'/(S+'.html')).as_uri());assert 'day 15 is excluded' in np.locator('body').inner_text();context.close();browser.close()
assert not errors,errors
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start()
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 for path,href in [('index.html','lessons/'+S+'.html'),('notebooks.html','labs/html/'+S+'.html')]:
  page.goto(f'http://127.0.0.1:{server.server_port}/'+path);reveal_gallery_link(page,'a[href="'+href+'"]')
 browser.close()
server.shutdown()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
count=0
for path in [R/'lessons'/(S+'.html'),R/'reference/gelgt-temporal-attention.html']:
 parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/gelgt-temporal-attention.html',P/(S+'.ipynb'),P/'evidence/l184/report.md']
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths];saved=(P/'solutions'/(S+'.ipynb')).read_bytes()
try:
 subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l184.py')],check=True,capture_output=True)
 assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
finally:(P/'solutions'/(S+'.ipynb')).write_bytes(saved)
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,python_javascript_parity='PASS',keyboard_reset='PASS',no_js='PASS',print='PASS',inline_source_parity='PASS',portable_figures=3,local_links=count,deterministic_build='PASS',manifest_galleries='PASS',javascript_errors=errors,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l184_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
