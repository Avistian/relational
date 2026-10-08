"""Validate portability, exact visible source, browser behavior and local navigation."""
import ast,functools,hashlib,itertools,json,os,subprocess,sys,tempfile,threading,zipfile
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
from relkit.landscape_l197 import admit_claim,essay_readiness
R=Path(__file__).resolve().parents[1];P=R/'labs';E=P/'evidence/l197';S='0197-year-5-essay'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
sc='\n\n'.join(c.source for c in student.cells if c.cell_type=='code');code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert sc.count('NotImplementedError')==3 and 'NotImplementedError' not in code
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert json.loads((P/'_execution_l197_results.json').read_text())['executed_code_sha256']==hashlib.sha256(code.encode()).hexdigest()
functions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
for path in [P/'relkit/landscape_l197.py',P/'_audit_l197.py',P/'_test_l197.py',E/'packet/relkit/tracking_l191.py',E/'packet/relkit/stress_l195.py',E/'packet/_replay_l191.py',E/'packet/_replay_l195.py']:
 for n in ast.parse(path.read_text()).body:
  if isinstance(n,ast.FunctionDef):assert functions[n.name]==ast.dump(n,include_attributes=False),n.name
paths=[R/'lessons'/(S+'.html'),R/'reference/year-5-essay.html',P/(S+'.ipynb'),E/'reproducer.zip',P/'figures/l197/landscape.svg']
before={p:p.read_bytes() for p in paths};executed=(P/'solutions'/(S+'.ipynb')).read_bytes()
subprocess.run([sys.executable,str(P/'_build_l197.py')],check=True,capture_output=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
rebuilt=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
assert '\n\n'.join(c.source for c in rebuilt.cells if c.cell_type=='code')==code
(P/'solutions'/(S+'.ipynb')).write_bytes(executed)
assert all(p.read_bytes()==b for p,b in before.items()),'Nondeterministic build'
# The CLI, as well as the inline notebook, must work outside the repository.
with tempfile.TemporaryDirectory(prefix='l197-zip-') as tmp:
 with zipfile.ZipFile(E/'reproducer.zip') as z:z.extractall(tmp)
 root=Path(tmp)
 subprocess.run([sys.executable,str(root/'labs/_audit_l197.py')],cwd=root,check=True,capture_output=True)
 assert json.loads((root/'labs/evidence/l197/report.json').read_text())==json.loads((E/'report.json').read_text())
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page(accept_downloads=True)
 page.on('pageerror',lambda error:errors.append(str(error)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto((R/'lessons'/(S+'.html')).as_uri())
  host=page.locator('#claim-audit')
  for lane,claim,auth,complete in itertools.product(['published_table','saved_predictions','source_diagnostic'],['published_comparison','scoped_pipeline_comparison','implementation_observation','fresh_model_reproduction','architecture_cause','economic_undervaluation'],[True,False],[True,False]):
   for selector,value in [('#essay-lane',lane),('#essay-claim',claim),('#essay-auth','yes' if auth else 'no'),('#essay-complete','yes' if complete else 'no')]:host.locator(selector).select_option(value)
   assert host.get_attribute('data-state')==admit_claim(lane,claim,auth,complete);states+=1
  host.locator('button').click();assert host.get_attribute('data-state')=='ADMISSIBLE_SCOPED'
  host.locator('#essay-claim').focus();page.keyboard.press('ArrowDown');assert host.get_attribute('data-state')=='NOT_ESTABLISHED'
  host.locator('button').click()
  figure=page.locator('#landscape-map')
  for strategy in ['graph','synthetic','reuse']:
   figure.locator('select').select_option(strategy);assert figure.get_attribute('data-strategy')==strategy;assert 'Ada' in figure.inner_text();states+=1
  figure.locator('button').click();assert figure.get_attribute('data-strategy')=='graph'
  # Existing HIN route behavior remains available after extension.
  assert page.evaluate('ArchFamilyViz.calculate("paper",false)[0][1]')==1
  essay=page.locator('#essay-structure');fields=['claim','evidence','warrant','limitation','revision']
  for bits in itertools.product([False,True],repeat=5):
   data={name:'One sentence.' if filled else ' ' for name,filled in zip(fields,bits)}
   for name,text in data.items():essay.locator('[name="'+name+'"]').fill(text)
   assert essay.get_attribute('data-state')==essay_readiness(data)['state'];states+=1
  with page.expect_download() as pending:essay.locator('button').click()
  download=pending.value;content=json.loads(Path(download.path()).read_text());assert content['review']['mastery']=='PENDING_WRITTEN_DEFENSE'
  for detail in page.locator('details').all():
   detail.locator('summary').focus();page.keyboard.press('Enter');assert detail.get_attribute('open') is not None
  assert 'A relational opportunity with unfinished proof' in page.locator('body').inner_text()
  assert not page.evaluate('document.documentElement.scrollWidth > innerWidth+1'),f'Overflow at {width}'
  page.evaluate('window.scrollTo(0,0)');page.screenshot(path=f'/tmp/l197-top-{width}.png')
  figure.screenshot(path=f'/tmp/l197-map-{width}.png');host.screenshot(path=f'/tmp/l197-audit-{width}.png')
  page.locator('.landscape-figure').screenshot(path=f'/tmp/l197-figure-{width}.png')
 page.emulate_media(media='print');assert host.locator('.controls').evaluate('(el)=>getComputedStyle(el).display')=='none'
 page.pdf(path='/tmp/l197-print.pdf',format='A4',print_background=True)
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri())
 assert 'Published tables support scoped published comparisons' in pg.locator('body').inner_text();assert not pg.evaluate('document.documentElement.scrollWidth > innerWidth+1');context.close()
 page.emulate_media(media='screen');page.set_viewport_size({'width':1200,'height':950});page.goto((P/'html'/(S+'.html')).as_uri())
 assert 'COMPLETE_SELECTED_EVIDENCE_AUDIT' in page.locator('body').inner_text()
 img=page.locator('img[alt="Three strategy routes"]');assert img.count()==1;assert img.evaluate('(el)=>el.complete && el.naturalWidth>0');img.screenshot(path='/tmp/l197-notebook-map.png')
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(SimpleHTTPRequestHandler,directory=str(R)))
 threading.Thread(target=server.serve_forever,daemon=True).start()
 try:
  for name in ['index.html','notebooks.html']:
   page.goto(f'http://127.0.0.1:{server.server_port}/'+name);reveal_gallery_link(page,'a[href*="'+S+'"]')
 finally:server.shutdown()
 assert not errors,errors;browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
links=0
for name in ['lessons/'+S+'.html','reference/year-5-essay.html']:
 path=R/name;parser=Links();parser.feed(path.read_text());assert '[[' not in path.read_text()
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
result=dict(status='PASS',interactive_states=states,viewports=[1200,375],keyboard_reset_download_print_nojs='PASS',source_ast_parity='PASS',deterministic_builder='PASS',standalone_zip_cli='EXACT_REPORT',local_links=links,manifest_galleries='PASS',prior_hin_widget='PASS',live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l197_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
