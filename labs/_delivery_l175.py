"""Rendered lesson/notebook, all access states, source parity and reproducible builds."""
import ast,functools,hashlib,json,os,subprocess,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0175-zero-shot-evaluation'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
solcode='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');stucode='\n\n'.join(c.source for c in student.cells if c.cell_type=='code')
assert 'NotImplementedError' not in solcode and stucode.count('NotImplementedError')==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
soldefs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(solcode).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for filename in ['relkit/zero_shot_l175.py','_check_l175.py','sources/l175/upstream/rt/model.py','_run_l175.py']:
 for n in ast.parse((P/filename).read_text()).body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)):assert soldefs[n.name]==ast.dump(n,include_attributes=False),n.name
execution=json.loads((P/'_execution_l175_results.json').read_text());assert execution['executed_code_sha256']==hashlib.sha256(solcode.encode()).hexdigest()
for item in json.loads((P/'sources/l175/source-ledger.json').read_text())['sources']:assert hashlib.sha256((P/'sources/l175'/item['file']).read_bytes()).hexdigest()==item['sha256']
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda msg:errors.append(msg.text) if msg.type=='error' else None);page.on('requestfailed',lambda req:errors.append(req.url))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/(S+'.html')).as_uri());host=page.locator('#access-explorer')
  for bits in range(16):
   flags=[bool(bits&(1<<i)) for i in range(4)]
   for i,flag in enumerate(flags):host.locator('input').nth(i).set_checked(flag)
   g,v,c,t=flags;expected='INVALID_TEST_EXPOSURE' if t else 'TARGET_TRAINED' if g else 'NO_TARGET_GRADIENTS_WITH_LABEL_ACCESS' if v or c else 'STRICT_NO_TARGET_LABEL_ACCESS'
   assert host.get_attribute('data-state')==expected;assert 'Fixed baseline' in host.locator('[data-baseline]').inner_text();states+=1
  host.locator('[data-reset]').click();assert host.get_attribute('data-state')=='NO_TARGET_GRADIENTS_WITH_LABEL_ACCESS'
  host.locator('input').first.focus();page.keyboard.press('Space');assert host.get_attribute('data-state')=='TARGET_TRAINED';host.locator('[data-reset]').click()
  host.screenshot(path=f'/tmp/l175-access-{width}.png')
  pred=page.locator('#predict');assert pred.locator('.predict-reveal').is_disabled();assert len(set(len(x.split()) for x in pred.locator('.predict-option').all_inner_texts()))==1
  pred.locator('[data-value=invalid]').click();pred.locator('.predict-reveal').click();assert '30 days' in pred.locator('.predict-outcome').inner_text()
  assert page.locator('#warmup button').count()==0 and page.locator('#teachback textarea').count()==1
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===3&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),f'Overflow {width}'
  page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l175-top-{width}.png')
  for i in range(3):page.locator('figure').nth(i).screenshot(path=f'/tmp/l175-figure-{i}-{width}.png')
 page.emulate_media(media='print');assert host.locator('.zs-controls').evaluate('(x)=>getComputedStyle(x).display')=='none'
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert pg.locator('noscript').count()==1 and '385' in pg.locator('body').inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.set_viewport_size({'width':1050,'height':900});page.goto((P/'html'/(S+'.html')).as_uri())
 assert page.locator('img[src^="data:image/png"]').count()==4
 assert 'Expected stop:' in page.locator('body').inner_text();page.locator('img[src^="data:image/png"]').first.screenshot(path='/tmp/l175-notebook-figure.png');browser.close()
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
for path in [R/'lessons'/(S+'.html'),R/'reference/zero-shot-evaluation.html']:
 parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/zero-shot-evaluation.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'evidence/l175/report.md']+sorted((P/'figures/l175').iterdir())
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
for script in ['_figures_l175.py','_build_l175.py']:subprocess.run([str(R/'.venv/bin/python'),str(P/script)],check=True,capture_output=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'Nondeterministic artifacts'
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',no_js='PASS',print='PASS',inline_source_parity='PASS',portable_figures=4,local_links=count,deterministic_build='PASS',manifest_galleries='PASS',javascript_errors=errors,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l175_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
