"""B08 provenance, student tasks, browser states, geometry, links and navigation."""
import ast,hashlib,json,os,tempfile,time,http.server,threading,functools
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
P=Path(__file__).resolve().parent;R=P.parent;S='b08-structured-objectives';V=R/'reviews/lesson-b08';V.mkdir(exist_ok=True);start=time.monotonic()
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
sc='\n\n'.join(c.source for c in student.cells if c.cell_type=='code');code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert sc.count('NotImplementedError')==3 and 'NotImplementedError' not in code
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert json.loads((P/'_execution_b08_results.json').read_text())['executed_code_sha256']==hashlib.sha256(code.encode()).hexdigest()
funcs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for n in ast.parse((P/'relkit/limix_b08.py').read_text()).body:
 if isinstance(n,(ast.FunctionDef,ast.ClassDef)):assert funcs[n.name]==ast.dump(n,include_attributes=False),n.name
with tempfile.TemporaryDirectory(prefix='b08-blank-') as td:
 try:NotebookClient(student,timeout=120,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
 except CellExecutionError as exc:assert 'NotImplementedError' in str(exc) and 'Complete sample_visibility' in str(exc)
 else:raise AssertionError('Blank student passed')
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
   sample=page.locator('[data-b08=sample]');feature=page.locator('[data-b08=feature]');loss=page.locator('[data-b08=loss]')
   for q2 in [10,20,30]:
    sample.locator('input[type=range]').fill(str(q2));sample.locator('input[type=range]').dispatch_event('input')
    for illegal in [False,True]:
     sample.locator('input[type=checkbox]').set_checked(illegal)
     expected=(18+q2)/4 if illegal else 4
     assert float(sample.locator('output').get_attribute('data-value'))==expected
     assert sample.locator('td[data-allowed=true]').count()==(16 if illegal else 8);states+=1
   sample.get_by_role('button').click();assert sample.locator('output').get_attribute('data-value')=='4'
   sample.locator('input[type=range]').focus();page.keyboard.press('ArrowLeft');assert sample.locator('input[type=range]').input_value()=='20';sample.get_by_role('button').click()
   for illegal in [False,True]:
    feature.locator('input[type=checkbox]').set_checked(illegal);assert abs(float(feature.locator('output').get_attribute('data-value'))-(28/3 if illegal else 4))<1e-12
    assert feature.locator('td[data-allowed=true]').count()==(9 if illegal else 8);states+=1
   feature.get_by_role('button').click()
   for weight in [0,.5,1,1.5,2]:
    loss.locator('input[type=range]').fill(str(weight));loss.locator('input[type=range]').dispatch_event('input')
    for all_cells in [False,True]:
     loss.locator('input[type=checkbox]').set_checked(all_cells)
     assert float(loss.locator('output').get_attribute('data-total'))==1+weight*(2 if all_cells else 2.5);states+=1
   loss.get_by_role('button').click();assert loss.locator('output').get_attribute('data-total')=='3.5'
   prediction=page.locator('#b08-predict');assert prediction.locator('.predict-reveal').is_disabled();prediction.locator('.predict-option').first.click();prediction.locator('.predict-reveal').click();assert '0.7536' in prediction.inner_text()
   tb=page.locator('#b08-teachback');assert tb.locator('button').first.is_disabled();tb.locator('textarea').fill('Support cannot read queries. Task slots read features. Loss changes paired gradients. Original row and cell identities are still missing, so the course does not establish historical reproduction.');tb.locator('button').first.click();assert tb.locator('input[type=checkbox]').count()==4
   assert page.locator('#b08-warmup button').count()>0
   page.locator('#b08-results summary').click();assert page.locator('#b08-results table').count()==1
   assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),width
   for img in page.locator('article img').all():
    img.scroll_into_view_if_needed();assert img.evaluate('(el)=>el.complete&&el.naturalWidth>0')
   page.evaluate('scrollTo(0,0)');page.screenshot(path=str(V/f'top-{width}.png'))
   for name,board in [('sample',sample),('feature',feature),('loss',loss)]:board.screenshot(path=str(V/f'{name}-{width}.png'))
   for i,name in enumerate(['limix16m','limix2','masks','results','paired']):page.locator('.b08-figure').nth(i).screenshot(path=str(V/f'{name}-{width}.png'))
  page.locator('#b08-results').evaluate("e=>e.removeAttribute('open')");page.emulate_media(media='print');assert page.locator('#b08-results table').is_visible();assert sample.locator('input').first.evaluate('(el)=>getComputedStyle(el).display')=='none';page.emulate_media(media='screen')
  ctx=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});pg=ctx.new_page();pg.goto(base+'lessons/'+S+'.html');assert '=4' in pg.locator('[data-b08=sample] output').inner_text();pg.locator('#b08-results summary').click();assert pg.locator('#b08-results table').count()==1;assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');ctx.close()
  page.goto(base+'index.html');page.locator('#lesson-nav a[href="lessons/'+S+'.html"]').wait_for()
  page.goto(base+'notebooks.html');page.locator('#lab-B08').wait_for();assert page.locator('#nb-list li').first.get_attribute('id')=='lab-B08'
  page.goto(base+'labs/html/'+S+'.html');assert page.locator('img').count()==4
  for img in page.locator('img').all():assert img.evaluate('(el)=>el.complete&&el.naturalWidth>0')
  page.set_viewport_size({'width':1000,'height':1100});page.locator('img').nth(1).screenshot(path=str(V/'notebook-limix2.png'))
  for name in ['limix16m','limix2']:
   page.goto(base+'labs/figures/b08/'+name+'.svg')
   bad=page.evaluate('''()=>Array.from(document.querySelectorAll('text')).map(e=>({text:e.textContent,b:e.getBBox()})).filter(o=>o.b.x<0||o.b.y<0||o.b.x+o.b.width>440||o.b.y+o.b.height>document.querySelector('svg').viewBox.baseVal.height)''');assert not bad,(name,bad)
  browser.close()
finally:server.shutdown()
assert not errors,errors
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
links=0
for path in [R/'lessons'/(S+'.html'),R/'reference'/(S+'.html')]:
 parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
result=dict(status='PASS',student='INTENTIONAL_TODO_FAILURE',canonical_AST='PASS',desktop_mobile_states=states,keyboard_reset='PASS',print_noJS='PASS',geometry='PASS',local_links=links,notebook_figures=4,browser_errors=errors,seconds=time.monotonic()-start,live_deployment='NOT_RUN',live_colab='NOT_CHECKED')
(P/'_delivery_b08_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
