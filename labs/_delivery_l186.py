"""Browser, notebook/source, deterministic-build and manifest navigation checks."""
import ast,functools,hashlib,itertools,json,os,subprocess,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0186-production-constraints'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
solcode='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');stucode='\n\n'.join(c.source for c in student.cells if c.cell_type=='code')
assert 'NotImplementedError' not in solcode and stucode.count('NotImplementedError')==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
defs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(solcode).body if isinstance(n,ast.FunctionDef)}
for filename in ['relkit/serving_l186.py','_check_l186.py','_audit_l186.py']:
 for n in ast.parse((P/filename).read_text()).body:
  if isinstance(n,ast.FunctionDef):assert defs[n.name]==ast.dump(n,include_attributes=False),(filename,n.name)
execution=json.loads((P/'_execution_l186_results.json').read_text());assert execution['executed_code_sha256']==hashlib.sha256(solcode.encode()).hexdigest()
markdown='\n'.join(c.source for c in solution.cells if c.cell_type=='markdown')
assert (P/'evidence/l186/packet/_run_l176.py').read_text() in markdown
r=json.loads((P/'evidence/l186/report.json').read_text());lookup={(x['policy'],str(x['rate']),x['condition'],str(x['seed'])):x for x in r['results']}
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda msg:errors.append(msg.text) if msg.type=='error' else None)
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':1000});page.goto((R/'lessons'/(S+'.html')).as_uri());host=page.locator('#serving-explorer')
  for (policy,rate,condition,seed),row in lookup.items():
   for field,value in [('policy',policy),('rate',rate),('condition',condition),('seed',seed)]:host.locator('[data-'+field+']').select_option(value)
   assert host.get_attribute('data-state')=='/'.join([policy,rate,condition,seed])
   out=host.locator('output');assert int(out.get_attribute('data-p99'))==row['p99_ms'];assert int(out.get_attribute('data-stale'))==row['stale_responses']
   if row['active_at_onset']:assert 'already active' in out.inner_text()
   assert str(row['labels_available_at_last_response'])+'/10000' in out.inner_text();states+=1
  host.locator('[data-reset]').click();assert host.get_attribute('data-state')=='cached/50/normal/0'
  host.locator('[data-rate]').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert host.get_attribute('data-state')=='cached/100/normal/0'
  host.locator('[data-reset]').click();host.screenshot(path=f'/tmp/l186-explorer-{width}.png')
  pred=page.locator('#predict');assert pred.locator('.predict-reveal').is_disabled();assert len(set(len(x.split()) for x in pred.locator('.predict-option').all_inner_texts()))==1
  pred.locator('[data-value=separate]').click();pred.locator('.predict-reveal').click();assert 'violates' in pred.locator('.predict-outcome').inner_text()
  assert page.locator('#warmup button').count()==0 and page.locator('#teachback textarea').count()==1
  page.locator('figure img').evaluate_all("xs=>xs.forEach(x=>x.loading='eager')")
  page.wait_for_function("[...document.querySelectorAll('figure img')].every(x=>x.complete&&x.naturalWidth>0)")
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===4&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),f'Overflow {width}'
  page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l186-top-{width}.png')
  for i in range(4):page.locator('figure').nth(i).screenshot(path=f'/tmp/l186-figure-{i}-{width}.png')
 page.emulate_media(media='print');assert host.locator('.sc-controls').evaluate('(x)=>getComputedStyle(x).display')=='none'
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert pg.locator('noscript').count()==1 and '810,000 responses' in pg.locator('body').inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.set_viewport_size({'width':1050,'height':900});page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==3
 assert 'COMPLETE_COURSE_SIMULATION' in page.locator('body').inner_text();page.locator('img[src^="data:image/png"]').first.screenshot(path='/tmp/l186-notebook-figure.png');browser.close()
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
for path in [R/'lessons'/(S+'.html'),R/'reference/production-constraints.html']:
 parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/production-constraints.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'evidence/l186/report.md',R/'assets/l186-evidence.js']+sorted((P/'figures/l186').iterdir())
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
for script in ['_figures_l186.py','_build_l186.py']:subprocess.run([str(R/'.venv/bin/python'),str(P/script)],check=True,capture_output=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'Nondeterministic artifacts'
result=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,report_javascript_parity='PASS',keyboard_reset='PASS',no_js='PASS',print='PASS',inline_source_parity='PASS',portable_figures=3,local_links=count,deterministic_build='PASS',manifest_galleries='PASS',javascript_errors=errors,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l186_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
