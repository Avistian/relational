"""Notebook provenance, blank student, source refusal, interactive/browser delivery."""
import ast,hashlib,json,os,tempfile,subprocess,sys,time,http.server,threading,functools
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
P=Path(__file__).resolve().parent;R=P.parent;S='b07a-hypernetworks';V=R/'reviews/lesson-b07a';V.mkdir(exist_ok=True);start=time.monotonic()
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
sc='\n\n'.join(c.source for c in student.cells if c.cell_type=='code');code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert sc.count('NotImplementedError')==3 and 'NotImplementedError' not in code
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert json.loads((P/'_execution_b07a_results.json').read_text())['executed_code_sha256']==hashlib.sha256(code.encode()).hexdigest()
funcs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for path in ['relkit/hyper_b07a.py','relkit/hyperfast_b07a.py','relkit/serving_b07a.py','_run_b07a.py','_audit_b07a.py','_source_b07a.py']:
 for n in ast.parse((P/path).read_text()).body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)):assert funcs[n.name]==ast.dump(n,include_attributes=False),n.name
with tempfile.TemporaryDirectory(prefix='b07a-blank-') as td:
 try:NotebookClient(student,timeout=120,kernel_name='python3',resources={'metadata':{'path':td}}).execute()
 except CellExecutionError as exc:assert 'NotImplementedError' in str(exc) and 'TODO: class_weights' in str(exc)
 else:raise AssertionError('Blank student passed')
result=subprocess.run([sys.executable,str(P/'_reproduce_b07a.py'),'--run'],capture_output=True,text=True)
assert result.returncode!=0 and 'INCOMPLETE_SOURCE_PROTOCOL' in result.stderr
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*a):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base='http://127.0.0.1:'+str(server.server_port)+'/'
errors=[];states=0
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.on('response',lambda r:errors.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
  for width in [1200,375]:
   page.set_viewport_size({'width':width,'height':950});page.goto(base+'lessons/'+S+'.html')
   board=page.locator('[data-b07a=weights]')
   for t in [-1,0,1,2,3]:
    board.locator('input[type=range]').fill(str(t));board.locator('input[type=range]').dispatch_event('input')
    for on in [False,True]:
     board.locator('input[type=checkbox]').set_checked(on)
     assert board.locator('output').get_attribute('data-logits')==f'{3+4*t+.2+(10 if on else 0):.1f},{6+7*t+.5:.1f}';states+=1
   board.get_by_role('button').click();assert board.locator('output').get_attribute('data-logits')=='11.2,20.5'
   board.locator('input[type=range]').focus();page.keyboard.press('ArrowLeft');assert board.locator('input[type=range]').input_value()=='1';board.get_by_role('button').click()
   cost=page.locator('[data-b07a=costs]')
   for u in [0,1,2]:
    cost.locator('select').select_option(str(u));assert cost.locator('output').get_attribute('data-threshold')==str(501+500*u);states+=1
   cost.locator('[data-volume]').fill('-9');cost.locator('[data-volume]').dispatch_event('input');assert cost.locator('[data-volume]').input_value()=='0'
   cost.get_by_role('button').click()
   prediction=page.locator('#b07a-predict');assert prediction.locator('.predict-reveal').is_disabled();prediction.locator('.predict-option').first.click();prediction.locator('.predict-reveal').click();assert 'iLTM' in prediction.inner_text()
   tb=page.locator('#b07a-teachback');assert tb.locator('button').first.is_disabled();tb.locator('textarea').fill('Generation uses support to create parameters. Retrieval still uses context. Count preprocessing, rebuilds and queries. Original split and search IDs remain missing.');tb.locator('button').first.click();assert tb.locator('input[type=checkbox]').count()==4
   assert page.locator('#b07a-warmup button').count()>0
   page.locator('#b07a-results summary').click();assert page.locator('#b07a-results table').count()==1
   assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),width
   for img in page.locator('article img').all():assert img.evaluate('(el)=>el.complete&&el.naturalWidth>0')
   page.evaluate('scrollTo(0,0)');page.screenshot(path=str(V/f'top-{width}.png'))
   board.screenshot(path=str(V/f'weights-{width}.png'));cost.screenshot(path=str(V/f'costs-{width}.png'))
   for i,name in enumerate(['mothernet','hyperfast','iltm','quality','timing']):page.locator('.b07a-figure').nth(i).screenshot(path=str(V/f'{name}-{width}.png'))
  page.locator('#b07a-results').evaluate("e=>e.removeAttribute('open')");page.emulate_media(media='print');assert page.locator('#b07a-results table').is_visible();assert board.locator('input').first.evaluate('(el)=>getComputedStyle(el).display')=='none';page.emulate_media(media='screen')
  ctx=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});pg=ctx.new_page();pg.goto(base+'lessons/'+S+'.html');assert '11.2' in pg.locator('[data-b07a=weights] output').inner_text();pg.locator('#b07a-results summary').click();assert pg.locator('#b07a-results table').count()==1;assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');ctx.close()
  page.goto(base+'index.html');page.locator('#lesson-nav a[href="lessons/'+S+'.html"]').wait_for()
  page.goto(base+'notebooks.html');page.locator('#lab-B07a').wait_for();assert page.locator('#nb-list li').first.get_attribute('id')=='lab-B07a'
  page.goto(base+'labs/html/'+S+'.html');assert page.locator('img').count()==5
  for img in page.locator('img').all():assert img.evaluate('(el)=>el.complete&&el.naturalWidth>0')
  page.set_viewport_size({'width':1000,'height':1100});page.locator('img').nth(1).screenshot(path=str(V/'notebook-hyperfast.png'))
  for name in ['mothernet','hyperfast','iltm']:
   page.goto(base+'labs/figures/b07a/'+name+'.svg')
   bad=page.evaluate('''()=>Array.from(document.querySelectorAll('text')).map(e=>({text:e.textContent,b:e.getBBox()})).filter(o=>o.b.x<0||o.b.y<0||o.b.x+o.b.width>460||o.b.y+o.b.height>document.querySelector('svg').viewBox.baseVal.height)''')
   assert not bad,(name,bad)
   boxes=page.evaluate("()=>Array.from(document.querySelectorAll('rect')).slice(1).map(e=>({x:+e.getAttribute('x'),y:+e.getAttribute('y'),w:+e.getAttribute('width'),h:+e.getAttribute('height')}))")
   for i,a in enumerate(boxes):
    assert a['x']>=0 and a['x']+a['w']<=460
    for b in boxes[i+1:]:assert not (a['x']<b['x']+b['w'] and a['x']+a['w']>b['x'] and a['y']<b['y']+b['h'] and a['y']+a['h']>b['y'])
  browser.close()
finally:server.shutdown()
assert not errors,errors
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
links=0
for path in [R/'lessons'/(S+'.html'),R/'reference/b07a-hypernetworks.html']:
 parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
result=dict(status='PASS',student='INTENTIONAL_TODO_FAILURE',canonical_AST='PASS',source_refusal='PASS',desktop_mobile_states=states,keyboard_reset='PASS',print_noJS='PASS',geometry='PASS',local_links=links,notebook_figures=5,browser_errors=errors,seconds=time.monotonic()-start,live_deployment='NOT_RUN',live_colab='NOT_CHECKED')
(P/'_delivery_b07a_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
