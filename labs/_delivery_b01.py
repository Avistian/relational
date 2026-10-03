"""Notebook parity, browser interaction and local links; no live deployment claim."""
import ast,functools,hashlib,http.server,itertools,json,os,subprocess,sys,tempfile,threading,zipfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from relkit.comparison_b01 import compare_contracts,claim_gate,FIELDS
R=Path(__file__).resolve().parents[1];P=R/'labs';E=P/'evidence/b01';S='b01-architecture-coverage-honest-comparison';V=R/'reviews/lesson-b01';V.mkdir(exist_ok=True)
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution_path=P/'solutions'/(S+'.ipynb');solution=nbformat.read(solution_path,4)
sc='\n\n'.join(c.source for c in student.cells if c.cell_type=='code');code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert sc.count('NotImplementedError')==3 and 'NotImplementedError' not in code
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert json.loads((P/'_execution_b01_results.json').read_text())['executed_code_sha256']==hashlib.sha256(code.encode()).hexdigest()
functions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
for name in ['relkit/comparison_b01.py','_audit_b01.py','_test_b01.py']:
 for n in ast.parse((P/name).read_text()).body:
  if isinstance(n,ast.FunctionDef):assert functions[n.name]==ast.dump(n,include_attributes=False),n.name
paths=[R/'lessons'/(S+'.html'),R/'reference/b01-comparison-contract.html',P/(S+'.ipynb'),E/'reproducer.zip',P/'figures/b01/information-flow.svg']
before={p:p.read_bytes() for p in paths};executed=solution_path.read_bytes()
try:
 subprocess.run([sys.executable,str(P/'_build_b01.py')],check=True,capture_output=True)
 rebuilt=nbformat.read(solution_path,4);assert '\n\n'.join(c.source for c in rebuilt.cells if c.cell_type=='code')==code
 assert all(p.read_bytes()==b for p,b in before.items()),'Non-deterministic builder'
finally:solution_path.write_bytes(executed)
with tempfile.TemporaryDirectory(prefix='b01-archive-') as td:
 with zipfile.ZipFile(E/'reproducer.zip') as z:z.extractall(td)
 subprocess.run([sys.executable,str(Path(td)/'labs/_verify_b01.py')],cwd=td,check=True,capture_output=True,timeout=120)
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*a):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)))
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();base='http://127.0.0.1:'+str(server.server_port)+'/'
errors=[];states=0
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.on('response',lambda r:errors.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
  for width in [1200,375]:
   page.set_viewport_size({'width':width,'height':950});page.goto(base+'lessons/'+S+'.html')
   board=page.locator('.comparison-board')
   for features,support,visibility,evidence,claim in itertools.product(['same','different','unknown'],['same','different','unknown'],['same','different','unknown'],['COMPLETE_REPLAY','INCOMPLETE'],['selected_score','architecture_cause','fresh_inference','learner_mastery']):
    vals=dict(features=features,support=support,visibility=visibility,evidence=evidence,claim=claim)
    # Dispatch normal change events while batching assignments to reduce test overhead.
    board.evaluate('(el,v)=>{Object.entries(v).forEach(([k,x])=>{let s=el.querySelector("[name="+k+"]");s.value=x;s.dispatchEvent(new Event("change",{bubbles:true}));});}',vals)
    a={k:'frozen' for k in FIELDS};b=dict(a)
    for k in ['features','support','visibility']:b[k]={'same':'frozen','different':'different','unknown':'UNKNOWN'}[vals[k]]
    expected=compare_contracts(a,b)['status'];out=board.locator('output')
    assert out.get_attribute('data-status')==expected
    assert out.get_attribute('data-claim')==claim_gate(expected,evidence,claim)
    states+=1
   board.get_by_role('button').click();assert board.locator('output').get_attribute('data-claim')=='SUPPORTED_DESCRIPTIVE_REPLAY'
   select=board.locator('select').first;select.focus();page.keyboard.press('ArrowDown');assert select.input_value()=='different';board.get_by_role('button').click()
   families=json.loads((E/'families.json').read_text())
   for f in families:
    page.get_by_label('Architecture family').select_option(f['id']);card=page.locator('[data-family="'+f['id']+'"]')
    assert card.is_visible() and f['example'] in card.inner_text()
    assert page.locator('[data-family]:visible').count()==1
   page.get_by_label('Architecture family').select_option('flat')
   assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),width
   assert page.locator('.b01-flow img').evaluate('(el)=>el.complete&&el.naturalWidth>0')
   page.evaluate('scrollTo(0,0)');page.screenshot(path=str(V/f'top-{width}.png'))
   board.screenshot(path=str(V/f'contract-{width}.png'));page.locator('.family-explorer').screenshot(path=str(V/f'family-{width}.png'));page.locator('.b01-flow').screenshot(path=str(V/f'flow-{width}.png'))
  page.emulate_media(media='print');assert board.locator('.comparison-controls').evaluate('(el)=>getComputedStyle(el).display')=='none'
  page.locator('.b01-flow').screenshot(path=str(V/'print-flow.png'));page.emulate_media(media='screen')
  context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});pg=context.new_page();pg.goto(base+'lessons/'+S+'.html')
  assert pg.locator('[data-family]:visible').count()==10 and 'SUPPORTED_DESCRIPTIVE_REPLAY' in pg.locator('output').inner_text()
  assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
  page.goto(base+'index.html');page.locator('#lesson-nav a[href="lessons/'+S+'.html"]').wait_for()
  assert 'LESSON B01' in page.locator('#lesson-nav').inner_text() and 'YEAR 5 → 6 BRIDGE' in page.locator('#lesson-nav').inner_text()
  assert 'LESSON 0200' in page.locator('#lesson-nav').inner_text()
  page.goto(base+'notebooks.html');page.locator('#lab-B01').wait_for();assert page.locator('#nb-list li').first.get_attribute('id')=='lab-B01'
  page.goto(base+'labs/html/'+S+'.html');assert 'COMPLETE_REPLAY' in page.locator('body').inner_text()
  assert page.locator('img').first.evaluate('(el)=>el.complete&&el.naturalWidth>0')
  page.locator('img').first.screenshot(path=str(V/'notebook-flow.png'))
  browser.close()
finally:server.shutdown();server.server_close()
assert not errors,errors
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
links=0
for path in paths[:2]:
 parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
r=dict(status='PASS',desktop_mobile_contract_states=states,family_states=20,keyboard_reset='PASS',print_nojs='PASS',portable_source_parity='PASS',deterministic_builder='PASS',archive_execution='PASS',manifest_galleries='PASS',local_links=links,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_b01_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
