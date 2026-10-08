"""Browser, portability, source and deterministic build verification."""
import ast,copy,functools,hashlib,itertools,json,os,subprocess,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
from relkit.comparison_l178 import comparison_gate
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0178-fair-model-comparison'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
solcode='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');stucode='\n\n'.join(c.source for c in student.cells if c.cell_type=='code')
assert 'NotImplementedError' not in solcode and stucode.count('NotImplementedError')==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
soldefs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(solcode).body if isinstance(n,ast.FunctionDef)}
for file in ['relkit/comparison_l178.py','_check_l178.py','_audit_l178.py']:
 for n in ast.parse((P/file).read_text()).body:
  if isinstance(n,ast.FunctionDef):assert soldefs[n.name]==ast.dump(n,include_attributes=False),(file,n.name)
markdown='\n'.join(c.source for c in solution.cells if c.cell_type=='markdown')
for file in ['sources/l166/upstream/model_pretrain/src/models.py','sources/l178/relgnn/examples__relgnn_nn.py','sources/l178/relgnn/examples__relgnn_conv.py','sources/l178/relgnn/examples__relgnn_model.py','sources/l178/rdblearn/rdblearn/estimator.py','sources/l178/rdblearn/rdblearn/preprocessing.py','_gradient_preflight_l178.py','_verify_gradient_l178.py','_preflight_l178.py']:assert (P/file).read_text() in markdown,file
execution=json.loads((P/'_execution_l178_results.json').read_text());assert execution['executed_code_sha256']==hashlib.sha256(solcode.encode()).hexdigest()
report=json.loads((P/'evidence/l178/report.json').read_text());errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('requestfailed',lambda q:errors.append(q.url))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':1000});page.goto((R/'lessons'/(S+'.html')).as_uri());host=page.locator('#comparison-explorer')
  for horizon,access,selection,audit,health in itertools.product(['30','60'],['matched','extra'],['validation','test'],['unknown','pass'],['fail','pass']):
   for key,value in [('horizon',horizon),('access',access),('selection',selection),('audit',audit),('health',health)]:host.locator('[data-'+key+']').select_option(value)
   contracts=copy.deepcopy(report['fresh']['contracts']);contracts['RDBLearn']['horizon_days']=int(horizon)
   contracts['RDBLearn']['target_history']='NONE' if access=='matched' else 'ALL_TRAIN'
   contracts['RelGNN']['selection']='VALIDATION_ONLY' if selection=='validation' else 'TEST'
   contracts['RelGNN']['training_health']='PASS' if health=='pass' else 'FAIL_NONFINITE_GRADIENT'
   for c in contracts.values():c['temporal_audit']=c['preprocessing_audit']='PASS' if audit=='pass' else 'NOT_CHECKED'
   expected=comparison_gate(contracts)['status'];assert host.locator('output').get_attribute('data-verdict')==expected
   assert host.get_attribute('data-state')=='/'.join([horizon,access,selection,audit,health]);states+=1
  host.locator('[data-reset]').click();assert host.get_attribute('data-state')=='30/matched/validation/unknown/fail'
  host.locator('[data-horizon]').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert host.locator('[data-horizon]').input_value()=='60'
  host.locator('[data-reset]').click();host.screenshot(path=f'/tmp/l178-explorer-{width}.png')
  pred=page.locator('#predict');assert pred.locator('.predict-reveal').is_disabled();assert len(set(len(x.split()) for x in pred.locator('.predict-option').all_inner_texts()))==1
  pred.locator('[data-value=both]').click();pred.locator('.predict-reveal').click();assert 'both' in pred.locator('.predict-outcome').inner_text()
  assert page.locator('#warmup button').count()==0 and page.locator('#teachback textarea').count()==1
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===3&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),f'Overflow {width}'
  page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l178-top-{width}.png')
  for i in range(3):page.locator('figure').nth(i).screenshot(path=f'/tmp/l178-figure-{i}-{width}.png')
 page.emulate_media(media='print');assert host.locator('.cg-controls').evaluate('(x)=>getComputedStyle(x).display')=='none'
 page.screenshot(path='/tmp/l178-print.png')
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert pg.locator('noscript').count()==1 and '21,060' in pg.locator('body').inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.set_viewport_size({'width':1050,'height':900});page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==3
 assert 'COMPLETE_SELECTED_PUBLISHED_REPLAY' in page.locator('body').inner_text();page.locator('img[src^="data:image/png"]').first.screenshot(path='/tmp/l178-notebook-figure.png');browser.close()
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
for path in [R/'lessons'/(S+'.html'),R/'reference/fair-model-comparison.html']:
 parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/fair-model-comparison.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'evidence/l178/report.md']+sorted((P/'figures/l178').iterdir())
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
for script in ['_figures_l178.py','_build_l178.py']:subprocess.run([str(R/'.venv/bin/python'),str(P/script)],check=True,capture_output=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'Nondeterministic artifacts'
result=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,python_javascript_verdict_parity='PASS',keyboard_reset='PASS',no_js='PASS',print='PASS',inline_source_parity='PASS',portable_figures=3,local_links=count,deterministic_build='PASS',manifest_galleries='PASS',javascript_errors=errors,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l178_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
