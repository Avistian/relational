"""Browser, portability, source, notebook and full-protocol admission verification."""
import ast,functools,hashlib,itertools,json,os,re,subprocess,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
from relkit.pretraining_l183 import factorial_effect
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0183-graph-transformer-pretraining'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
solcode='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');stucode='\n\n'.join(c.source for c in student.cells if c.cell_type=='code')
assert stucode.count('NotImplementedError')==4 and 'NotImplementedError' not in solcode
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None for c in solution.cells if c.cell_type=='code')
soldefs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(solcode).body if isinstance(n,ast.FunctionDef)}
for file,names in [('relkit/pretraining_l183.py',None),('_check_l183.py',None),('_audit_l183.py',None),('_verify_l183.py',['sql_mae'])]:
 for n in ast.parse((P/file).read_text()).body:
  if isinstance(n,ast.FunctionDef) and (names is None or n.name in names):assert soldefs[n.name]==ast.dump(n,include_attributes=False),(file,n.name)
markdown='\n'.join(c.source for c in solution.cells if c.cell_type=='markdown')
source_paths=['labs/relkit/relgt_l145.py','labs/_full_l145.py','labs/relkit/griffin_l164.py','labs/sources/l164/upstream/hmodel.py','labs/sources/l164/upstream/hmaintask_downsample_absolute_eval_sample.py','labs/_run_l164.py']
for i,name in enumerate(source_paths):
 block=markdown.split('### '+name+'\nSHA256:',1)[1]
 if i+1<len(source_paths):block=block.split('### '+source_paths[i+1]+'\nSHA256:',1)[0]
 reconstructed=''.join(re.findall(r'```python\n(.*?)\n```',block,re.S))
 assert reconstructed==(R/name).read_text(),name
execution=json.loads((P/'_execution_l183_results.json').read_text());assert execution['executed_code_sha256']==hashlib.sha256(solcode.encode()).hexdigest()
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('requestfailed',lambda q:errors.append(q.url))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':1000});page.goto((R/'lessons'/(S+'.html')).as_uri());host=page.locator('#factorial')
  for preset,expected in [('extra',.3),('equal',0.),('best',-.5)]:
   host.locator('[data-preset]').select_option(preset);assert abs(float(host.locator('output').get_attribute('data-interaction'))-expected)<1e-9;states+=1
  for values in itertools.product([0.,3.,6.],repeat=4):
   host.locator('input').evaluate_all('(xs,vs)=>{xs.forEach((x,i)=>{x.value=vs[i];});xs[0].dispatchEvent(new Event("input",{bubbles:true}));}',list(values))
   arms={k:{0:v,1:v} for k,v in zip(['mp_scratch','mp_pretrained','gt_scratch','gt_pretrained'],values)}
   assert abs(float(host.locator('output').get_attribute('data-interaction'))-factorial_effect(arms)['interaction_mean'])<1e-9;states+=1
  host.locator('[data-reset]').click();assert abs(float(host.locator('output').get_attribute('data-interaction'))-.3)<1e-9
  host.locator('input').first.focus();page.keyboard.press('ArrowRight');assert host.locator('input').first.input_value()=='4.1'
  host.locator('[data-reset]').click();host.screenshot(path=f'/tmp/l183-explorer-{width}.png')
  pred=page.locator('#predict');assert pred.locator('.predict-reveal').is_disabled();assert len(set(len(x.split()) for x in pred.locator('.predict-option').all_inner_texts()))==1
  pred.locator('[data-value=gnn]').click();pred.locator('.predict-reveal').click();assert 'course GNN' in pred.locator('.predict-outcome').inner_text()
  assert page.locator('#warmup button').count()==0 and page.locator('#teachback textarea').count()==1
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===5&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),f'Overflow {width}'
  page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l183-top-{width}.png')
  for i in range(5):
   fig=page.locator('figure').nth(i);fig.screenshot(path=f'/tmp/l183-figure-{i}-{width}.png')
   if width==375:fig.evaluate('(x)=>x.scrollLeft=x.scrollWidth');fig.screenshot(path=f'/tmp/l183-figure-{i}-{width}-end.png')
 page.emulate_media(media='print');assert host.locator('.fc-controls').evaluate('(x)=>getComputedStyle(x).display')=='none';page.screenshot(path='/tmp/l183-print.png')
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert pg.locator('noscript').count()==1 and '7,554' in pg.locator('body').inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.set_viewport_size({'width':1050,'height':900});page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==6
 assert 'COMPLETE_SELECTED_SAVED_PREDICTION_REPLAY' in page.locator('body').inner_text();page.locator('img[src^="data:image/png"]').first.screenshot(path='/tmp/l183-notebook-figure.png');browser.close()
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
for path in [R/'lessons'/(S+'.html'),R/'reference/graph-transformer-pretraining.html']:
 parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/graph-transformer-pretraining.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'evidence/l183/report.md']+sorted((P/'figures/l183').iterdir())
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
for script in ['_figures_l183.py','_build_l183.py']:subprocess.run([str(R/'.venv/bin/python'),str(P/script)],check=True,capture_output=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'Nondeterministic artifacts'
for lane in ['relgt','griffin']:
 out=subprocess.run([str(R/'.venv/bin/python'),str(P/'_run_l183.py'),'--lane',lane,'--run'],capture_output=True,text=True)
 assert out.returncode!=0 and 'No training dispatched' in out.stderr
result=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,python_javascript_parity='PASS',keyboard_reset='PASS',no_js='PASS',print='PASS',inline_source_parity='PASS',complete_model_trainer_appendices=6,portable_figures=6,local_links=count,deterministic_build='PASS',manifest_galleries='PASS',full_protocol_admission='BOTH_REFUSE_TRAINING',javascript_errors=errors,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l183_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
