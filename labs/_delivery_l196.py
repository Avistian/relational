"""Validate artifact consistency, actual browser behavior and manifest navigation."""
import ast,functools,hashlib,itertools,json,os,subprocess,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
from relkit.community_l196 import feedback_state
R=Path(__file__).resolve().parents[1];P=R/'labs';E=P/'evidence/l196';S='0196-community-engagement'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
stucode='\n\n'.join(c.source for c in student.cells if c.cell_type=='code')
solcode='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert stucode.count('NotImplementedError')==3 and 'NotImplementedError' not in solcode
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert json.loads((P/'_execution_l196_results.json').read_text())['executed_code_sha256']==hashlib.sha256(solcode.encode()).hexdigest()
defs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(solcode).body if isinstance(n,ast.FunctionDef)}
for path in [P/'relkit/community_l196.py',P/'_diagnostic_l196.py',P/'_report_l196.py',P/'_test_l196.py']:
 for node in ast.parse(path.read_text()).body:
  if isinstance(node,ast.FunctionDef):assert defs[node.name]==ast.dump(node,include_attributes=False),node.name
# Regeneration is deterministic; keep the executed solution bytes and HTML output.
paths=[R/'lessons'/(S+'.html'),R/'reference/community-engagement.html',P/(S+'.ipynb'),E/'reproducer.zip',E/'question-draft.md']
before={p:p.read_bytes() for p in paths};executed=(P/'solutions'/(S+'.ipynb')).read_bytes()
subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l196.py')],cwd=R,check=True,capture_output=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
rebuilt=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
assert '\n\n'.join(c.source for c in rebuilt.cells if c.cell_type=='code')==solcode
(P/'solutions'/(S+'.ipynb')).write_bytes(executed)
assert all(p.read_bytes()==b for p,b in before.items()),'Nondeterministic builder'
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 page.on('pageerror',lambda error:errors.append(str(error)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950})
  page.goto((R/'lessons'/(S+'.html')).as_uri())
  host=page.locator('#community-feedback')
  for posted,replied,verified in itertools.product([False,True],repeat=3):
   for selector,value in zip(['#community-post','#community-reply','#community-check'],[posted,replied,verified]):
    host.locator(selector).select_option('yes' if value else 'no')
   expected=feedback_state('https://example.org/thread/1' if posted else '', 'actual reply' if replied else '',verified)
   assert host.get_attribute('data-state')==expected
   assert 'actual record remains DRAFT_ONLY' in host.inner_text()
   states+=1
  host.locator('button').click();assert host.get_attribute('data-state')=='DRAFT_ONLY'
  host.locator('#community-post').focus();page.keyboard.press('ArrowDown')
  assert host.get_attribute('data-state')=='AWAITING_RESPONSE'
  for detail in page.locator('details').all():
   detail.locator('summary').focus();page.keyboard.press('Enter');assert detail.get_attribute('open') is not None
  assert not page.evaluate('document.documentElement.scrollWidth > innerWidth+1'),f'Overflow {width}'
  page.evaluate('window.scrollTo(0,0)');page.screenshot(path=f'/tmp/l196-top-{width}.png')
  host.screenshot(path=f'/tmp/l196-widget-{width}.png')
  page.locator('.repro-table').first.screenshot(path=f'/tmp/l196-table-{width}.png')
 page.emulate_media(media='print')
 assert host.locator('.controls').evaluate('(el)=>getComputedStyle(el).display')=='none'
 page.pdf(path='/tmp/l196-print.pdf',format='A4',print_background=True)
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950})
 pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri())
 assert 'Feedback states: no thread' in pg.locator('body').inner_text()
 assert not pg.evaluate('document.documentElement.scrollWidth > innerWidth+1')
 context.close()
 page.emulate_media(media='screen');page.goto((P/'html'/(S+'.html')).as_uri())
 assert 'Original implementation appendix' in page.locator('body').inner_text()
 assert 'COMPLETE_DIAGNOSTIC' in page.locator('body').inner_text()
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(SimpleHTTPRequestHandler,directory=str(R)))
 threading.Thread(target=server.serve_forever,daemon=True).start()
 try:
  for name in ['index.html','notebooks.html']:
   page.goto(f'http://127.0.0.1:{server.server_port}/'+name)
   reveal_gallery_link(page,'a[href*="'+S+'"]')
 finally:server.shutdown()
 assert not errors,errors
 browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
links=0
for name in ['lessons/'+S+'.html','reference/community-engagement.html']:
 path=R/name;parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url
  links+=1
result=dict(status='PASS',interactive_states=states,viewports=[1200,375],keyboard_reset_print_nojs='PASS',source_ast_parity='PASS',deterministic_builder='PASS',local_links=links,manifest_galleries='PASS',live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l196_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
