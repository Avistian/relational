"""Check notebook source, standalone CLI, local links and interactive decisions."""
import ast,hashlib,itertools,json,os,subprocess,sys,tempfile,zipfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from relkit.direction_l199 import priority,launch_gate,memo_readiness
R=Path(__file__).resolve().parents[1];P=R/'labs';E=P/'evidence/l199';S='0199-select-primary-direction'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
sc='\n\n'.join(c.source for c in student.cells if c.cell_type=='code');code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert sc.count('NotImplementedError')==3 and 'NotImplementedError' not in code
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert json.loads((P/'_execution_l199_results.json').read_text())['executed_code_sha256']==hashlib.sha256(code.encode()).hexdigest()
functions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
for path in [P/'relkit/direction_l199.py',P/'_audit_l199.py',P/'_test_l199.py',E/'packet/relkit/checkpoint_l190.py',E/'packet/_replay_l190.py']:
 for n in ast.parse(path.read_text()).body:
  if isinstance(n,ast.FunctionDef):assert functions[n.name]==ast.dump(n,include_attributes=False),n.name
paths=[R/'lessons'/(S+'.html'),R/'reference/select-primary-direction.html',P/(S+'.ipynb'),E/'reproducer.zip',P/'figures/l199/decision.svg']
before={p:p.read_bytes() for p in paths};executed=(P/'solutions'/(S+'.ipynb')).read_bytes()
subprocess.run([sys.executable,str(P/'_build_l199.py')],check=True,capture_output=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
rebuilt=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
assert '\n\n'.join(c.source for c in rebuilt.cells if c.cell_type=='code')==code
(P/'solutions'/(S+'.ipynb')).write_bytes(executed)
assert all(p.read_bytes()==b for p,b in before.items()),'Nondeterministic build'
with tempfile.TemporaryDirectory(prefix='l199-zip-') as tmp:
 with zipfile.ZipFile(E/'reproducer.zip') as z:z.extractall(tmp)
 root=Path(tmp);subprocess.run([sys.executable,str(root/'labs/_audit_l199.py')],cwd=root,check=True,capture_output=True)
 assert json.loads((root/'labs/evidence/l199/report.json').read_text())==json.loads((E/'report.json').read_text())
cases=json.loads((E/'packet/evidence/l190/packet/cases.json').read_text());errors=[];states=0;V=R/'reviews/lesson-199';V.mkdir(exist_ok=True)
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page(accept_downloads=True)
 page.on('pageerror',lambda error:errors.append(str(error)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto((R/'lessons'/(S+'.html')).as_uri())
  ranking=page.locator('#direction-ranking');gates=page.locator('#direction-gates');memo=page.locator('#direction-memo')
  for impact in range(1,6):
   cc=json.loads(json.dumps(cases));cc[0]['impact']=impact;ranking.locator('#rank-impact').select_option(str(impact))
   for w in itertools.product([1,2,3],repeat=3):
    for k,v in zip(['data','implementation','compute'],w):ranking.locator('#rank-'+k).select_option(str(v))
    assert ranking.get_attribute('data-leaders')==','.join(priority(cc,list(w))['leaders']);states+=1
  ranking.locator('button').click();assert ranking.get_attribute('data-leaders')=='availability'
  ranking.locator('#rank-impact').focus();page.keyboard.press('ArrowUp');assert ranking.get_attribute('data-leaders')=='composite'
  ranking.locator('button').click()
  for values in itertools.product(['PASS','FAIL','UNKNOWN'],repeat=4):
   data=dict(zip(['data','baseline','design','budget'],values))
   for k,v in data.items():gates.locator('#gate-'+k).select_option(v)
   assert gates.get_attribute('data-state')==launch_gate(data)['state'];states+=1
  gates.locator('button').click();assert gates.get_attribute('data-state')=='DO_NOT_LAUNCH'
  fields=['direction','hypothesis','contrast','baselines','metric','threshold','uncertainty','cost','stop','deferred','evidence','revision']
  for k in fields:memo.locator('[name="'+k+'"]').fill('A reason requiring human review.')
  assert memo.get_attribute('data-state')=='READY_FOR_REVIEW'
  for k in fields:
   memo.locator('[name="'+k+'"]').fill(' ');assert memo.get_attribute('data-state')=='DRAFT';states+=1
   memo.locator('[name="'+k+'"]').fill('A reason requiring human review.')
  with page.expect_download() as pending:memo.locator('button').first.click()
  content=json.loads(Path(pending.value.path()).read_text());assert content['review']['mastery']=='PENDING_WRITTEN_DEFENSE' and content['authorization']=='NOT_GRANTED'
  memo.locator('button').nth(1).click();assert memo.get_attribute('data-state')=='DRAFT'
  for d in page.locator('details').all():d.locator('summary').focus();page.keyboard.press('Enter');assert d.get_attribute('open') is not None
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),f'Page overflow {width}'
  page.evaluate('window.scrollTo(0,0)');page.screenshot(path=str(V/f'top-{width}.png'))
  ranking.screenshot(path=str(V/f'ranking-{width}.png'));gates.screenshot(path=str(V/f'gates-{width}.png'));page.locator('.direction-figure').screenshot(path=str(V/f'figure-{width}.png'))
 page.emulate_media(media='print');assert gates.locator('.direction-controls').evaluate('(el)=>getComputedStyle(el).display')=='none'
 page.screenshot(path=str(V/'print.png'),full_page=False)
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri())
 assert 'All four current requirements are UNKNOWN' in pg.locator('body').inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.set_viewport_size({'width':1200,'height':950});page.goto((P/'html'/(S+'.html')).as_uri())
 assert 'COMPLETE_SELECTED_EVIDENCE_AUDIT' in page.locator('body').inner_text()
 img=page.locator('img[alt="Research decision flow"]');assert img.count()==1 and img.evaluate('(el)=>el.complete && el.naturalWidth>0');img.screenshot(path=str(V/'notebook-figure.png'))
 assert not errors,errors;browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
links=0
for name in ['lessons/'+S+'.html','reference/select-primary-direction.html']:
 path=R/name;parser=Links();parser.feed(path.read_text());assert '[[' not in path.read_text()
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
result=dict(status='PASS',browser_states=states,widths=[1200,375],keyboard_reset_export='PASS',no_javascript_print='PASS',local_links=links,visible_source_parity='EXACT',standalone_zip='PASS',deterministic_build='PASS',console_errors=errors,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l199_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
