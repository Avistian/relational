"""Browser arithmetic, portable-code parity, links and deterministic build."""
import ast,functools,hashlib,itertools,json,os,re,subprocess,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
from relkit.gaps_l189 import priority
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0189-identify-open-problems'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
sc='\n\n'.join(c.source for c in student.cells if c.cell_type=='code');code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert sc.count('raise NotImplementedError')==3 and 'raise NotImplementedError' not in code
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
soldefs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
for file in ['relkit/gaps_l189.py','_audit_l189.py','_test_l189.py']:
 for n in ast.parse((P/file).read_text()).body:
  if isinstance(n,ast.FunctionDef):assert soldefs[n.name]==ast.dump(n,include_attributes=False),n.name
assert sum(c.source.count('data:image/png;base64,') for c in student.cells)==2
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':1000});page.goto((R/'lessons'/(S+'.html')).as_uri());host=page.locator('#priority')
  for weights in itertools.product([1,2,3],repeat=3):
   for impact in [1,2,3,4,5]:
    for i,v in enumerate(list(weights)+[impact]):host.locator('[data-control="'+str(i)+'"]').evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input"));}',v)
    state=json.loads(host.get_attribute('data-result'));expected={'temporal':priority(impact,[5,4,5],weights),'composite':priority(5,[4,3,2],weights),'transfer':priority(5,[3,2,1],weights)}
    for row in state['scores']:assert abs(row['score']-expected[row['id']])<1e-12
    assert state['leaders']==sorted(k for k,v in expected.items() if v==max(expected.values()));states+=1
  host.locator('[data-reset]').click();assert json.loads(host.get_attribute('data-result'))['impact']==4
  host.locator('[data-control="0"]').focus();page.keyboard.press('ArrowRight');assert json.loads(host.get_attribute('data-result'))['weights'][0]==2
  host.locator('[data-reset]').click();host.screenshot(path=f'/tmp/l189-widget-{width}.png')
  page.evaluate('window.scrollTo(0,0)');page.screenshot(path=f'/tmp/l189-lesson-{width}.png')
  for box in page.locator('#checklist input').all():box.check()
  assert 'Proposal structure checked' in page.locator('#checklist').inner_text()
  page.locator('details').first.locator('summary').click();assert page.locator('details').first.get_attribute('open') is not None
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Page overflow'
  fig=page.locator('figure').first
  if width==375:assert fig.locator('img').bounding_box()['width']<=width
  fig.screenshot(path=f'/tmp/l189-trace-{width}.png')
 page.emulate_media(media='print');assert page.locator('#priority .rp-controls').is_hidden()
 page.pdf(path='/tmp/l189-print.pdf',format='A4',print_background=True)
 nojs=browser.new_context(java_script_enabled=False);np=nojs.new_page();np.goto((R/'lessons'/(S+'.html')).as_uri())
 assert 'Equal weights: temporal 18.67' in np.inner_text('body');assert np.locator('details').count()==4
 np.locator('details').nth(1).locator('summary').click();assert 'Minimum full comparison' in np.locator('details').nth(1).inner_text()
 page.emulate_media(media='screen');page.set_viewport_size({'width':1000,'height':1000});page.goto((P/'html'/(S+'.html')).as_uri());page.screenshot(path='/tmp/l189-notebook.png')
 assert page.locator('img[src^="data:image/png"]').count()==2
 browser.close()
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
for path in [R/'lessons'/(S+'.html'),R/'reference/identify-open-problems.html']:
 parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;count+=1
for path in [P/'l189-reproduction.md',P/'evidence/l189/report.md',P/'evidence/l189/shortlist.md']:
 for url in re.findall(r'\]\(([^)]+)\)',path.read_text()):
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),(path,url);count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/identify-open-problems.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'evidence/l189/report.md',P/'evidence/l189/shortlist.md']+sorted((P/'figures/l189').iterdir())
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
for script in ['_figures_l189.py','_build_l189.py']:subprocess.run([str(R/'.venv/bin/python'),str(P/script)],check=True,capture_output=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'Nondeterministic artifacts'
result=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,python_javascript_parity='PASS',keyboard_reset='PASS',no_js='PASS',print='PASS',inline_source_parity='PASS',portable_figures=2,local_links=count,deterministic_build='PASS',manifest_galleries='PASS',javascript_errors=errors,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l189_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
