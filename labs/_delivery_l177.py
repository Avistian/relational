"""Check all calculator states against Python, portability and publication links."""
import ast,functools,hashlib,itertools,json,os,subprocess,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
from relkit.compute_l177 import assess_plan
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0177-compute-budget-realism'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
solcode='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');stucode='\n\n'.join(c.source for c in student.cells if c.cell_type=='code')
assert 'NotImplementedError' not in solcode and stucode.count('NotImplementedError')==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
soldefs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(solcode).body if isinstance(n,ast.FunctionDef)}
for filename in ['relkit/compute_l177.py','_check_l177.py','_audit_l177.py']:
 for n in ast.parse((P/filename).read_text()).body:
  if isinstance(n,ast.FunctionDef):assert soldefs[n.name]==ast.dump(n,include_attributes=False),(filename,n.name)
markdown='\n'.join(c.source for c in solution.cells if c.cell_type=='markdown')
for name in ['_run_l173.py','_run_l174.py','_run_l176.py','sources/l177/modal-l176.py','_dispatch_l175.py']:assert (P/name).read_text() in markdown,name
execution=json.loads((P/'_execution_l177_results.json').read_text());assert execution['executed_code_sha256']==hashlib.sha256(solcode.encode()).hexdigest()
r=json.loads((P/'evidence/l177/report.json').read_text());errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda msg:errors.append(msg.text) if msg.type=='error' else None);page.on('requestfailed',lambda req:errors.append(req.url))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':1000});page.goto((R/'lessons'/(S+'.html')).as_uri());host=page.locator('#budget-explorer')
  for key,minutes,active,mem,device in itertools.product(['forecast','observed','pretrain','finetune','stopped'],[60,180],['unknown','1200','3600'],['unknown','16','32'],[24,8]):
   for field,value in [('case',key),('session',minutes),('active',active),('memory',mem),('device',device)]:host.locator('[data-'+field+']').select_option(str(value))
   a=None if active=='unknown' else int(active);m=None if mem=='unknown' else int(mem)
   if key in ['forecast','observed']:cost=r['l176']['reserved_usd'];wall=r['l176']['forecast_seconds' if key=='forecast' else 'worker_seconds']+120
   elif key=='stopped':cost=r['l175']['reserved_usd'];wall=930
   else:row=r['rt_price_scenarios']['pretraining' if key=='pretrain' else 'fine_tuning'];cost=row['gpu_only_usd_40gb'];wall=row['reported_elapsed_hours']*3600
   expected=assess_plan(cost,wall,a,m,device,minutes*60,key!='stopped')['status']
   if key in ['forecast','observed'] and device<r['l176']['max_allocator_gib'] and expected!='OVER_BUDGET':expected='MEMORY_LIMIT'
   assert host.locator('output').get_attribute('data-verdict')==expected,(key,minutes,active,mem,device)
   assert f'${cost:.6f}' in host.locator('output').inner_text()
   assert host.get_attribute('data-state')=='/'.join(map(str,[key,minutes,active,mem,device]));states+=1
  host.locator('[data-reset]').click();assert host.get_attribute('data-state')=='forecast/60/unknown/unknown/24'
  host.locator('[data-session]').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert host.get_attribute('data-state')=='forecast/180/unknown/unknown/24'
  host.locator('[data-active]').select_option('1200');host.locator('[data-memory]').select_option('16');assert host.locator('output').get_attribute('data-verdict')=='FEASIBLE_SCENARIO'
  host.screenshot(path=f'/tmp/l177-explorer-{width}.png')
  pred=page.locator('#predict');assert pred.locator('.predict-reveal').is_disabled();assert len(set(len(x.split()) for x in pred.locator('.predict-option').all_inner_texts()))==1
  pred.locator('[data-value=outer]').click();pred.locator('.predict-reveal').click();assert 'double-counts' in pred.locator('.predict-outcome').inner_text()
  assert page.locator('#warmup button').count()>0 and page.locator('#teachback textarea').count()==1
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===3&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),f'Overflow {width}'
  page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l177-top-{width}.png')
  for i in range(3):page.locator('figure').nth(i).screenshot(path=f'/tmp/l177-figure-{i}-{width}.png')
 page.emulate_media(media='print');assert host.locator('.cb-controls').evaluate('(x)=>getComputedStyle(x).display')=='none'
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert pg.locator('noscript').count()==1 and '18 fit records' in pg.locator('body').inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.set_viewport_size({'width':1050,'height':900});page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==3
 assert 'COMPLETE_SELECTED_ACCOUNTING_REPLAY' in page.locator('body').inner_text();page.locator('img[src^="data:image/png"]').first.screenshot(path='/tmp/l177-notebook-figure.png');browser.close()
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
for path in [R/'lessons'/(S+'.html'),R/'reference/compute-budget-realism.html']:
 parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/compute-budget-realism.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'evidence/l177/report.md',R/'assets/l177-evidence.js']+sorted((P/'figures/l177').iterdir())
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
for script in ['_figures_l177.py','_build_l177.py']:subprocess.run([str(R/'.venv/bin/python'),str(P/script)],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'Nondeterministic artifacts'
result=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,python_javascript_verdict_parity='PASS',keyboard_reset='PASS',no_js='PASS',print='PASS',inline_source_parity='PASS',portable_figures=3,local_links=count,deterministic_build='PASS',manifest_galleries='PASS',javascript_errors=errors,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l177_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
