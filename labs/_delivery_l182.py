"""Actual browser/portable-source/independent-arithmetic and deterministic build checks."""
import ast,functools,hashlib,itertools,json,os,subprocess,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
import numpy as np
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
from relkit.composite_l182 import attend
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0182-rdb-pfn-composite-message-passing'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
solcode='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');stucode='\n\n'.join(c.source for c in student.cells if c.cell_type=='code')
assert 'NotImplementedError' not in solcode and stucode.count('NotImplementedError')==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
soldefs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(solcode).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for file,names in [('relkit/composite_l182.py',['legal_fusion','attend','keyed_auc','factorial_interaction']),('_report_l182.py',['summarize182']),('_mechanism_l182.py',['mechanism182']),('_check_l182.py',['checks']),('relkit/rdbpfn_l166.py',['normalize_support','FeatureEncoder','TargetEncoder','BiAttention','Decoder','RDBPFN'])]:
 for n in ast.parse((P/file).read_text()).body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names:assert soldefs[n.name]==ast.dump(n,include_attributes=False),n.name
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':1000});page.goto((R/'lessons'/(S+'.html')).as_uri());host=page.locator('#composite-route')
  for dup,time,mix,rev in itertools.product(range(3),[9,11],[False,True],[False,True]):
   host.locator('[data-duplicates]').evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input"));}',dup)
   host.locator('[data-time]').select_option(str(time))
   host.locator('[data-mix]').set_checked(mix);host.locator('[data-reverse]').set_checked(rev)
   state=json.loads(host.get_attribute('data-result'))
   messages=[[1,1],[2,1]]+[[1,1]]*dup
   if time==9:messages.append([100,99])
   if mix:messages.append([8,0])
   if rev:messages.reverse()
   expected,weights=attend(np.array([[1.,0.]]),np.array(messages,float),np.zeros(len(messages),dtype=int))
   np.testing.assert_allclose(state['output'],expected[0],atol=1e-12);np.testing.assert_allclose(state['weights'],weights,atol=1e-12);states+=1
  host.locator('[data-reset]').click();assert json.loads(host.get_attribute('data-result'))['output'][0]==1.6697615493266569
  host.locator('[data-duplicates]').focus();page.keyboard.press('ArrowRight');assert json.loads(host.get_attribute('data-result'))['duplicates']==1
  host.screenshot(path=f'/tmp/l182-duplicate-{width}.png');host.locator('[data-reset]').click();host.screenshot(path=f'/tmp/l182-explorer-{width}.png')
  pred=page.locator('#predict');assert pred.locator('.predict-reveal').is_disabled();assert len(set(len(x.split()) for x in pred.locator('.predict-option').all_inner_texts()))==1
  pred.locator('[data-value=change]').click();pred.locator('.predict-reveal').click();assert 'mass' in pred.locator('.predict-outcome').inner_text()
  assert page.locator('#warmup button').count()==0 and page.locator('#teachback textarea').count()==1
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===4&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),f'Overflow {width}'
  page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l182-top-{width}.png')
  for i in range(4):page.locator('figure').nth(i).screenshot(path=f'/tmp/l182-figure-{i}-{width}.png')
 page.emulate_media(media='print');assert host.locator('.cr-controls').evaluate('(x)=>getComputedStyle(x).display')=='none';page.screenshot(path='/tmp/l182-print.png')
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert pg.locator('noscript').count()==1 and '21,060' in pg.locator('body').inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.set_viewport_size({'width':1050,'height':900});page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==4
 assert 'COMPLETE_SELECTED_REPRODUCTION' in page.locator('body').inner_text();page.locator('img[src^="data:image/png"]').nth(0).screenshot(path='/tmp/l182-notebook-figure.png');browser.close()
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
for path in [R/'lessons'/(S+'.html'),R/'reference/rdb-pfn-composite-message-passing.html']:
 parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/rdb-pfn-composite-message-passing.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'evidence/l182/report.md']+sorted((P/'figures/l182').iterdir())
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
for script in ['_figures_l182.py','_build_l182.py']:subprocess.run([str(R/'.venv/bin/python'),str(P/script)],check=True,capture_output=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'Nondeterministic artifacts'
result=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,python_javascript_arithmetic_parity='PASS',keyboard_reset='PASS',no_js='PASS',print='PASS',inline_source_parity='PASS',portable_figures=4,local_links=count,deterministic_build='PASS',manifest_galleries='PASS',javascript_errors=errors,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l182_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
