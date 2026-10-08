"""Real browser checks, portable source parity, local links and deterministic build."""
import ast,functools,hashlib,itertools,json,os,subprocess,threading
from pathlib import Path
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
from html.parser import HTMLParser
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
R=Path(__file__).resolve().parents[1];P=R/'labs';E=P/'evidence/l191';S='0191-kumorfm2-sota-tracking'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
sc='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');st='\n\n'.join(c.source for c in student.cells if c.cell_type=='code')
assert 'NotImplementedError' not in sc and st.count('NotImplementedError')==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None for c in solution.cells if c.cell_type=='code')
defs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(sc).body if isinstance(n,ast.FunctionDef)}
for path in [P/'relkit/tracking_l191.py',P/'_replay_l191.py']:
 for n in ast.parse(path.read_text()).body:
  if isinstance(n,ast.FunctionDef):assert defs[n.name]==ast.dump(n,include_attributes=False),n.name
assert hashlib.sha256(sc.encode()).hexdigest()==json.loads((P/'_execution_l191_results.json').read_text())['executed_code_sha256']
report=json.loads((E/'report.json').read_text());errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto((R/'lessons'/(S+'.html')).as_uri());host=page.locator('#sota-explorer')
  for t in report['tables']:
   for pool,rule in itertools.product(['foundation','supervised','all'],['single','oracle']):
    host.locator('[data-table]').select_option(str(t['number']));host.locator('[data-pool]').select_option(pool);host.locator('[data-rule]').select_option(rule);expected=t['pools'][pool]
    assert host.get_attribute('data-status')==expected['status']
    if expected['status']=='NO_ELIGIBLE_COMPARATOR':assert host.get_attribute('data-gap')=='' and host.locator('tbody tr').count()==0
    else:
     assert abs(float(host.get_attribute('data-gap'))-expected[rule+'_gap'])<1e-10
     assert host.locator('tbody tr').count()==len(t['tasks'])
     if rule=='single':assert host.get_attribute('data-methods')==','.join(expected['single_methods'])
    assert '+4.053333' in host.locator('.baseline').inner_text();states+=1
  host.locator('[data-reset]').click();assert abs(float(host.get_attribute('data-gap'))-4.05333333333334)<1e-10
  host.locator('[data-pool]').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert host.locator('[data-pool]').input_value()=='supervised';host.locator('[data-reset]').click()
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===2&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),f'Overflow {width}'
  assert page.locator('figure img').first.evaluate('(x)=>x.currentSrc').endswith('architecture-mobile.svg' if width==375 else 'architecture.svg')
  host.screenshot(path=f'/tmp/l191-widget-{width}.png');page.locator('figure').first.screenshot(path=f'/tmp/l191-architecture-{width}.png');page.locator('figure').nth(1).screenshot(path=f'/tmp/l191-gaps-{width}.png');page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l191-top-{width}.png')
  # Native details preserve all task data without horizontal page overflow.
  for detail in page.locator('details').all():detail.evaluate('(x)=>x.open=true')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
 page.emulate_media(media='print');assert host.locator('.controls').evaluate('(x)=>getComputedStyle(x).display')=='none'
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert '79.782' in pg.locator('body').inner_text();assert pg.locator('details').count()>=4;assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.set_viewport_size({'width':1000,'height':900});page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==2;page.locator('img[src^="data:image/png"]').first.screenshot(path='/tmp/l191-notebook.png')
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(SimpleHTTPRequestHandler,directory=str(R)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  for name in ['index.html','notebooks.html']:
   page.goto(f'http://127.0.0.1:{server.server_port}/'+name);reveal_gallery_link(page,'a[href*="'+S+'"]')
 finally:server.shutdown()
 assert not errors,errors;browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src','srcset'])
links=0
for name in ['lessons/'+S+'.html','reference/kumorfm2-sota-tracking.html']:
 path=R/name;parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
outputs=[R/'lessons'/(S+'.html'),R/'reference/kumorfm2-sota-tracking.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb')]+list((P/'figures/l191').glob('*'))
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}
for command in ['_figures_l191.py','_build_l191.py']:subprocess.run([str(R/'.venv/bin/python'),str(P/command)],check=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},'Nondeterministic builder'
result=dict(status='PASS',browser_states=states,widths=[1200,375],keyboard_reset=True,no_js=True,print_media=True,inline_source_parity=True,deterministic_build=True,local_links=links,galleries=True,portable_figures=2,mobile_figures=2,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l191_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
