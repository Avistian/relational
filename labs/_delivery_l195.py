"""Browser parity, portable notebook source, links and deterministic generation."""
import ast,functools,hashlib,json,os,subprocess,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
from relkit.stress_l195 import interval_verdict,claim_scope
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0195-thesis-stress-test'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
stucode='\n\n'.join(c.source for c in student.cells if c.cell_type=='code');solcode='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert stucode.count('NotImplementedError')==3 and 'NotImplementedError' not in solcode
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert json.loads((P/'_execution_l195_results.json').read_text())['executed_code_sha256']==hashlib.sha256(solcode.encode()).hexdigest()
defs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(solcode).body if isinstance(n,ast.FunctionDef)}
for path in [P/'relkit/stress_l195.py',P/'_replay_l195.py']:
 for n in ast.parse(path.read_text()).body:
  if isinstance(n,ast.FunctionDef):assert defs[n.name]==ast.dump(n,include_attributes=False),n.name
errors=[];states=0;ci=json.loads((P/'evidence/l195/report.json').read_text())['regression']['test']['conditional_interval']
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto((R/'lessons'/(S+'.html')).as_uri());host=page.locator('#margin-explorer')
  for complete in [True,False]:
   host.locator('select').select_option('yes' if complete else 'no')
   for k in range(51):
    host.locator('input').evaluate('(el,v)=>{el.value=v;el.dispatchEvent(new Event("input",{bubbles:true}))}',str(k/100));expected=interval_verdict(ci['low'],ci['high'],k/100,complete)
    assert host.get_attribute('data-verdict')==expected;assert '−0.174155' in host.inner_text();states+=1
  host.locator('[data-reset]').click();assert host.get_attribute('data-verdict')=='UNRESOLVED';host.locator('input').focus();page.keyboard.press('ArrowRight');assert host.locator('input').input_value()=='0.01'
  claims=page.locator('#claim-explorer')
  for packet in ['l149','l182','l194']:
   claims.locator('#stress-packet').select_option(packet)
   for claim in ['pipeline','relational_signal','architecture_cause','general_superiority','undervaluation','fresh_training']:
    claims.locator('#stress-claim').select_option(claim);e=dict(authenticated=True,complete=packet!='l194',measured=packet!='l194',comparable=True)
    assert claims.get_attribute('data-verdict')==claim_scope(claim,e);states+=1
  claims.locator('[data-reset]').click();assert claims.get_attribute('data-verdict')=='SCOPED_COMPARISON';claims.locator('#stress-claim').focus();page.keyboard.press('ArrowDown');assert claims.get_attribute('data-verdict')=='NOT_ESTABLISHED'
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===3&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),f'Overflow {width}'
  pred=page.locator('#prediction');pred.locator('button').first.click();pred.locator('button').last.click();assert 'Both systems use' in pred.inner_text()
  tb=page.locator('#teachback');tb.locator('textarea').fill('I narrow C2. The conditional interval crosses zero, and both pipelines use relational features. A new matched-information study could change this position.');tb.locator('button').first.click();assert 'practical margin' in tb.inner_text()
  page.evaluate('window.scrollTo(0,0)');page.screenshot(path=f'/tmp/l195-top-{width}.png');host.screenshot(path=f'/tmp/l195-margin-{width}.png');claims.screenshot(path=f'/tmp/l195-claim-{width}.png')
  for j in range(3):page.locator('figure').nth(j).screenshot(path=f'/tmp/l195-figure-{j}-{width}.png')
 page.emulate_media(media='print')
 for selector in ['#margin-explorer','#claim-explorer']:assert page.locator(selector+' .controls').evaluate('(x)=>getComputedStyle(x).display')=='none'
 page.pdf(path='/tmp/l195-print.pdf',format='A4',print_background=True)
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert '33,650' in pg.locator('body').inner_text();assert pg.locator('noscript').count()==3;assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.set_viewport_size({'width':1000,'height':950});page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==3;page.locator('img[src^="data:image/png"]').nth(1).screenshot(path='/tmp/l195-notebook.png')
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
for name in ['lessons/'+S+'.html','reference/thesis-stress-test.html']:
 path=R/name;parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
outputs=[R/'lessons'/(S+'.html'),R/'reference/thesis-stress-test.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb')]+list((P/'figures/l195').glob('*'))
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs};subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l195.py')],check=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},'Nondeterministic builder'
result=dict(status='PASS',browser_states=states,widths=[1200,375],keyboard_reset=True,no_js=True,print=True,inline_source_parity=True,deterministic_build=True,local_links=links,galleries=True,portable_figures=3,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l195_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
