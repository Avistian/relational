"""Browser behavior, notebook implementation and deterministic-build checks."""
import ast,functools,hashlib,json,os,subprocess,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0194-open-fm-analysis-report'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
stucode='\n\n'.join(c.source for c in student.cells if c.cell_type=='code');solcode='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert stucode.count('NotImplementedError')==3 and 'NotImplementedError' not in solcode
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert json.loads((P/'_execution_l194_results.json').read_text())['executed_code_sha256']==hashlib.sha256(solcode.encode()).hexdigest()
defs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(solcode).body if isinstance(n,ast.FunctionDef)}
for path in [P/'relkit/report_l194.py',P/'_replay_l194.py']:
 for n in ast.parse(path.read_text()).body:
  if isinstance(n,ast.FunctionDef):assert defs[n.name]==ast.dump(n,include_attributes=False),n.name
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto((R/'lessons'/(S+'.html')).as_uri());host=page.locator('#evidence-explorer')
  for kind,license in [('PUBLISHED_TABLE','DESCRIPTIVE_REFERENCE_ONLY'),('SAVED_SOURCE_DIAGNOSTIC','SOURCE_INVARIANT_FAILURE_ONLY'),('UNRUN_BENCHMARK','NO_PERFORMANCE_CONCLUSION'),('PROPOSED_INTERVENTION','HYPOTHESIS_NOT_TESTED')]:
   host.locator('select').select_option(kind);assert host.get_attribute('data-license')==license;assert '0/21' in host.inner_text();states+=1
  host.locator('[data-reset]').click();assert host.locator('select').input_value()=='PUBLISHED_TABLE'
  host.locator('select').focus();page.keyboard.press('ArrowDown');assert host.get_attribute('data-license')=='SOURCE_INVARIANT_FAILURE_ONLY'
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===2&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),f'Overflow {width}'
  pred=page.locator('#prediction');pred.locator('button').first.click();pred.locator('button').last.click();assert 'Both use relational' in pred.inner_text()
  tb=page.locator('#teachback');tb.locator('textarea').fill('We observed code changes; benchmark performance remains unmeasured. A controlled study must measure the score effect.');tb.locator('button').first.click();assert 'recorded preprocessing' in tb.inner_text().lower()
  page.screenshot(path=f'/tmp/l194-top-{width}.png');host.screenshot(path=f'/tmp/l194-evidence-{width}.png')
  for i in range(2):page.locator('figure').nth(i).screenshot(path=f'/tmp/l194-figure-{i}-{width}.png')
 page.emulate_media(media='print');assert host.locator('.controls').evaluate('(x)=>getComputedStyle(x).display')=='none'
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert '0/630' in pg.locator('body').inner_text();assert pg.locator('noscript').count()==3;assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==2;page.locator('img[src^="data:image/png"]').nth(1).screenshot(path='/tmp/l194-notebook.png')
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(SimpleHTTPRequestHandler,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start()
 try:
  for name in ['index.html','notebooks.html']:
   page.goto(f'http://127.0.0.1:{server.server_port}/'+name);reveal_gallery_link(page,'a[href*="'+S+'"]')
 finally:server.shutdown()
 assert not errors,errors;browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
links=0
for name in ['lessons/'+S+'.html','reference/open-fm-analysis-report.html']:
 path=R/name;parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
outputs=[R/'lessons'/(S+'.html'),R/'reference/open-fm-analysis-report.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb')]+list((P/'figures/l194').glob('*'))
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs};subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l194.py')],check=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},'Nondeterministic builder'
result=dict(status='PASS',browser_states=states,widths=[1200,375],keyboard_reset=True,no_js=True,print=True,inline_source_parity=True,deterministic_build=True,local_links=links,galleries=True,portable_figures=2,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l194_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
