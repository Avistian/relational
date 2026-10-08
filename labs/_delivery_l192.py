"""Real browser, source parity, links, portable figures and deterministic outputs."""
import ast,functools,hashlib,json,os,re,subprocess,threading
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0192-open-fm-setup-data'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
student_code='\n\n'.join(c.source for c in student.cells if c.cell_type=='code');solution_code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert student_code.count('raise NotImplementedError')==3
assert not any(c.get('outputs') for c in student.cells if c.cell_type=='code')
assert 'raise NotImplementedError' not in solution_code
nodes={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(solution_code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for name in ['relkit/setup_l192.py','_audit_l192.py','_test_l192.py']:
 for n in ast.parse((P/name).read_text()).body:
  if isinstance(n,ast.FunctionDef):assert ast.dump(n,include_attributes=False)==nodes[n.name],n.name
original=next(n for n in ast.parse((P/'sources/l192/rdblearn/rdblearn/preprocessing.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='SafeLabelEncoderTransformer')
assert nodes[original.name]==ast.dump(original,include_attributes=False)
assert sum(c.source.count('data:image/png;base64,') for c in student.cells)==4
rows=json.loads((P/'evidence/l192/packet/queries.json').read_text());assert len({r['entity'] for r in rows})==len(rows)==13779
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':1000});page.goto((R/'lessons'/(S+'.html')).as_uri())
  encoding=page.locator('#encoding');clock=page.locator('#clock')
  for policy in ['released','frozen']:
   for category in ['a','c','z']:
    encoding.locator('[data-category]').select_option(category);encoding.locator('[data-policy]').select_option(policy)
    result=json.loads(encoding.get_attribute('data-result'));expected=[1,2,3] if policy=='released' and category=='a' else [0,1,2]
    assert result['codes']==expected;states+=1
  for day in range(401):
   clock.locator('[data-day]').evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input"));}',day)
   result=json.loads(clock.get_attribute('data-result'));assert result==dict(day=day,released=day>0,available=day>=365);states+=1
  for host in [encoding,clock]:host.locator('[data-reset]').click()
  clock.locator('[data-day]').focus();page.keyboard.press('ArrowRight');assert json.loads(clock.get_attribute('data-result'))['day']==181
  clock.locator('[data-reset]').click();encoding.locator('[data-category]').focus();page.keyboard.press('ArrowDown');assert json.loads(encoding.get_attribute('data-result'))['category']=='c'
  encoding.locator('[data-reset]').click()
  assert page.locator('#warmup').count()==0
  for box in page.locator('#checklist input').all():box.check()
  assert 'gate remains failed' in page.locator('#checklist').inner_text()
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Horizontal page overflow'
  page.evaluate('window.scrollTo(0,0)');page.screenshot(path=f'/tmp/l192-top-{width}.png')
  for i,fig in enumerate(page.locator('figure').all()):
   assert fig.locator('img').evaluate('(e)=>e.complete&&e.naturalWidth>0')
   if width==375:assert '-mobile.svg' in fig.locator('img').evaluate('(e)=>e.currentSrc')
   fig.screenshot(path=f'/tmp/l192-figure-{i}-{width}.png')
  encoding.screenshot(path=f'/tmp/l192-encoding-{width}.png');clock.screenshot(path=f'/tmp/l192-clock-{width}.png')
 page.emulate_media(media='print');assert page.locator('#encoding .rc-controls').is_hidden();page.pdf(path='/tmp/l192-print.pdf',format='A4',print_background=True)
 nojs=browser.new_context(java_script_enabled=False);np=nojs.new_page();np.goto((R/'lessons'/(S+'.html')).as_uri());assert 'Intervene mentally' in np.inner_text('body')
 page.emulate_media(media='screen');page.set_viewport_size({'width':1000,'height':1000});page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==4
 page.locator('figure').nth(2).screenshot(path='/tmp/l192-notebook-figure.png');browser.close()
assert not errors,errors
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src','srcset'])
links=0
for path in [R/'lessons'/(S+'.html'),R/'reference/open-fm-setup-data.html']:
 parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),(path,url);links+=1
for url in re.findall(r'\]\(([^)]+)\)',(P/'l192-reproduction.md').read_text()):
 part=urlsplit(url)
 if not part.scheme and part.path:assert (P/unquote(part.path)).resolve().is_file(),url;links+=1
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start()
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 for path,href in [('index.html','lessons/'+S+'.html'),('notebooks.html','labs/html/'+S+'.html')]:
  page.goto(f'http://127.0.0.1:{server.server_port}/'+path);reveal_gallery_link(page,'a[href="'+href+'"]')
 browser.close()
server.shutdown()
paths=[R/'lessons'/(S+'.html'),R/'reference/open-fm-setup-data.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb')]+sorted((P/'figures/l192').iterdir())
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
for name in ['_figures_l192.py','_build_l192.py']:subprocess.run([str(R/'.venv/bin/python'),str(P/name)],check=True,capture_output=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'Nondeterministic build'
result=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',no_js='PASS',print='PASS',inline_source_parity='PASS',portable_figures=4,mobile_reflow_figures=3,local_links=links,deterministic_build='PASS',manifest_galleries='PASS',javascript_errors=errors,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l192_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
