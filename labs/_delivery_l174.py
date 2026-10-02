"""Browser interaction, inline source, links, gallery and deterministic build checks."""
import ast,functools,hashlib,json,os,subprocess,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0174-fine-tuning-protocol'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
solcode='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');stucode='\n\n'.join(c.source for c in student.cells if c.cell_type=='code')
assert 'NotImplementedError' not in solcode and stucode.count('NotImplementedError')==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
soldefs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(solcode).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for filename in ['relkit/multitask_l173.py','relkit/finetune_l174.py','_check_l174.py']:
 for n in ast.parse((P/filename).read_text()).body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and not (filename=='relkit/multitask_l173.py' and n.name in {'erase_target','load_packet','train_run'}):assert soldefs[n.name]==ast.dump(n,include_attributes=False),n.name
execution=json.loads((P/'_execution_l174_results.json').read_text());assert execution['executed_code_sha256']==hashlib.sha256(solcode.encode()).hexdigest()
manifest=json.loads((P/'evidence/l174/manifest.json').read_text())
for group,base in [('files',P/'evidence/l174'),('inputs',P),('source_files',P)]:
 for name,h in manifest[group].items():assert hashlib.sha256((base/name).read_bytes()).hexdigest()==h,name
for item in json.loads((P/'sources/l174/source-ledger.json').read_text())['sources']:assert hashlib.sha256((P/'sources/l174'/item['file']).read_bytes()).hexdigest()==item['sha256']
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda msg:errors.append(msg.text) if msg.type=='error' else None);page.on('requestfailed',lambda req:errors.append(req.url))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/(S+'.html')).as_uri());host=page.locator('#policy-explorer')
  for arm,count in [('freeze',9834),('full',28426),('adapter',10386),('scratch',28426)]:
   host.locator('select').select_option(arm);assert host.get_attribute('data-trainable')==str(count)
   assert host.locator('[data-block=encoder]').get_attribute('data-learn')==str(arm in ['full','scratch']).lower()
   assert 'Fixed baseline' in host.locator('[data-baseline]').inner_text();states+=1
  host.screenshot(path=f'/tmp/l174-policy-{width}.png');host.locator('[data-reset]').click();assert host.get_attribute('data-arm')=='freeze'
  host.locator('select').focus();page.keyboard.press('ArrowDown');assert host.get_attribute('data-arm')=='full';host.locator('[data-reset]').click()
  pred=page.locator('#predict');assert pred.locator('.predict-reveal').is_disabled()
  assert len(set(len(x.split()) for x in pred.locator('.predict-option').all_inner_texts()))==1
  pred.locator('[data-value=up]').click();pred.locator('.predict-reveal').click();assert 'down projection' in pred.locator('.predict-outcome').inner_text()
  assert page.locator('#warmup button').count()>0 and page.locator('#teachback textarea').count()==1
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===3&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),f'Overflow {width}'
  page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l174-top-{width}.png')
  for i in range(3):page.locator('figure').nth(i).screenshot(path=f'/tmp/l174-figure-{i}-{width}.png')
 page.emulate_media(media='print');assert host.locator('.finetune-controls').evaluate('(x)=>getComputedStyle(x).display')=='none'
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri())
 assert pg.locator('noscript').count()==1 and '106,757' in pg.locator('body').inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.set_viewport_size({'width':1050,'height':900});page.goto((P/'html'/(S+'.html')).as_uri())
 assert page.locator('img[src^="data:image/png"]').count()>=4
 assert 'Fresh selected epoch:' in page.locator('body').inner_text();page.locator('img[src^="data:image/png"]').first.screenshot(path='/tmp/l174-notebook-figure.png');browser.close()
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
for path in [R/'lessons'/(S+'.html'),R/'reference/fine-tuning-protocol.html']:
 parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/fine-tuning-protocol.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'evidence/l174/runs/report.md']+sorted((P/'figures/l174').iterdir())
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
for script in ['_figures_l174.py','_build_l174.py']:subprocess.run([str(R/'.venv/bin/python'),str(P/script)],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'Nondeterministic artifacts'
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',no_js='PASS',print='PASS',inline_source_parity='PASS',portable_figures=3,fresh_notebook_plot=True,local_links=count,deterministic_build='PASS',manifest_galleries='PASS',javascript_errors=errors,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l174_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
