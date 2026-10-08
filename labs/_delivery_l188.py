"""Verify portable code, exact executed report, browser states and deterministic builds."""
import ast,functools,hashlib,itertools,json,os,subprocess,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
from relkit.literature_l188 import triage
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0188-systematic-literature-tracking'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
solcode='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');stucode='\n\n'.join(c.source for c in student.cells if c.cell_type=='code')
assert 'NotImplementedError' not in solcode and stucode.count('NotImplementedError')==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
soldefs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(solcode).body if isinstance(n,ast.FunctionDef)}
for n in ast.parse((P/'relkit/literature_l188.py').read_text()).body:
 if isinstance(n,ast.FunctionDef):assert soldefs[n.name]==ast.dump(n,include_attributes=False),n.name
execution=json.loads((P/'_execution_l188_results.json').read_text());assert execution['executed_code_sha256']==hashlib.sha256(solcode.encode()).hexdigest()
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto((R/'lessons'/(S+'.html')).as_uri());host=page.locator('#triage')
  for verified,reviewed,role in itertools.product(['yes','no'],['yes','no'],['baseline','failure_mode','sota','incremental']):
   for key,val in [('verified',verified),('reviewed',reviewed),('role',role)]:host.locator('[data-'+key+']').select_option(val)
   expected=triage(dict(verified=verified=='yes',reviewed=reviewed=='yes',relevance=role,in_window=True,reason='Synthetic example',evidence='reported'))
   assert host.get_attribute('data-decision')==expected;assert 'NOT_RUN' in host.locator('.lit-baseline').inner_text();states+=1
  host.locator('[data-reset]').click();assert host.get_attribute('data-decision')=='INCLUDE'
  host.locator('[data-verified]').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert host.get_attribute('data-decision')=='DEFER'
  host.screenshot(path=f'/tmp/l188-triage-{width}.png');host.locator('[data-reset]').click()
  pred=page.locator('#predict');assert pred.locator('.predict-reveal').is_disabled()
  assert len(set(len(x.split()) for x in pred.locator('.predict-option').all_inner_texts()))==1
  pred.locator('[data-value=incomplete]').click();pred.locator('.predict-reveal').click();assert '103' in pred.locator('.predict-outcome').inner_text()
  assert page.locator('#warmup button').count()==0 and page.locator('#teachback textarea').count()==1
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===1&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),f'Overflow {width}'
  page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l188-top-{width}.png');page.locator('figure').screenshot(path=f'/tmp/l188-figure-{width}.png')
 page.emulate_media(media='print');assert host.locator('.lit-controls').evaluate('(x)=>getComputedStyle(x).display')=='none'
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert 'Thirty' not in pg.title();assert '30 unique' in pg.locator('body').inner_text();assert pg.locator('noscript').count()==1;assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.set_viewport_size({'width':1000,'height':900});page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==1;page.locator('img[src^="data:image/png"]').screenshot(path='/tmp/l188-notebook.png')
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(SimpleHTTPRequestHandler,directory=str(R)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  base=f'http://127.0.0.1:{server.server_port}'
  for name in ['index.html','notebooks.html']:
   page.goto(base+'/'+name);reveal_gallery_link(page,'a[href*="'+S+'"]');assert page.locator('a[href*="'+S+'"]').count()>0
 finally:server.shutdown()
 assert not errors,errors;browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
links=0
for name in ['lessons/'+S+'.html','reference/systematic-literature-tracking.html']:
 path=R/name;parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
outputs=[R/'lessons'/(S+'.html'),R/'reference/systematic-literature-tracking.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'figures/l188/evidence-flow.svg',P/'figures/l188/evidence-flow.png',P/'evidence/l188/paper-log.md']
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}
subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l188.py')],check=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},'Nondeterministic build'
result=dict(status='PASS',browser_states=states,widths=[1200,375],keyboard_reset=True,no_js=True,print=True,portable_figure=True,inline_source_parity=True,deterministic_build=True,local_links=links,galleries=True,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l188_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
