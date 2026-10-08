"""Verify portable source, reproducible build, browser states and publication links."""
import ast,hashlib,itertools,json,os,subprocess,sys,tempfile,zipfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from relkit.exit_l200 import exit_gate
R=Path(__file__).resolve().parents[1];P=R/'labs';E=P/'evidence/l200';S='0200-year-5-exit-exam';V=R/'reviews/lesson-200';V.mkdir(exist_ok=True)
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
sc='\n\n'.join(c.source for c in student.cells if c.cell_type=='code');code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert sc.count('NotImplementedError')==3 and 'NotImplementedError' not in code
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert json.loads((P/'_execution_l200_results.json').read_text())['executed_code_sha256']==hashlib.sha256(code.encode()).hexdigest()
functions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for path in [P/'relkit/exit_l200.py',P/'_audit_l200.py',P/'_test_l200.py',P/'_verify_l200.py',E/'packet/rdbpfn_visible.py']:
 for n in ast.parse(path.read_text()).body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)):assert functions[n.name]==ast.dump(n,include_attributes=False),n.name
paths=[R/'lessons'/(S+'.html'),R/'reference/year-5-exit-exam.html',P/(S+'.ipynb'),E/'reproducer.zip',P/'figures/l200/architecture.svg']
before={p:p.read_bytes() for p in paths};executed=(P/'solutions'/(S+'.ipynb')).read_bytes()
subprocess.run([sys.executable,str(P/'_build_l200.py')],check=True,capture_output=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
rebuilt=nbformat.read(P/'solutions'/(S+'.ipynb'),4);assert '\n\n'.join(c.source for c in rebuilt.cells if c.cell_type=='code')==code
(P/'solutions'/(S+'.ipynb')).write_bytes(executed)
assert all(p.read_bytes()==b for p,b in before.items()),'Nondeterministic build'
with tempfile.TemporaryDirectory(prefix='l200-zip-') as tmp:
 with zipfile.ZipFile(E/'reproducer.zip') as z:z.extractall(tmp)
 root=Path(tmp)
 for name in ['_audit_l200.py','_verify_l200.py']:subprocess.run([sys.executable,str(root/'labs'/name)],cwd=root,check=True,capture_output=True)
 assert json.loads((root/'labs/evidence/l200/report.json').read_text())==json.loads((E/'report.json').read_text())
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 page.on('pageerror',lambda error:errors.append(str(error)))
 page.on('requestfailed',lambda request:errors.append(request.url))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto((R/'lessons'/(S+'.html')).as_uri())
  board=page.locator('#exit200');selects=board.locator('select')
  assert 'proposal, defense' in board.locator('output').inner_text()
  for values in itertools.product(['PASS','FAIL','PENDING'],repeat=3):
   for i,value in enumerate(values):selects.nth(i).select_option(value)
   expected=exit_gate(*values);text=board.locator('output').inner_text();assert text.startswith(expected['state'])
   for blocker in expected['blockers']:assert blocker in text
   states+=1
  board.locator('button').click();assert [selects.nth(i).input_value() for i in range(3)]==['PASS','PENDING','PENDING']
  selects.nth(0).focus();page.keyboard.press('ArrowDown');assert selects.nth(0).input_value()=='FAIL';board.locator('button').click()
  assert page.locator('#warmup').count()==0
  predict=page.locator('#predict200');assert predict.locator('button').count()>0
  predict.locator('button').first.click()
  # Check prediction and teachback using public rendered controls.
  reveal=predict.get_by_role('button',name='Reveal',exact=False)
  if reveal.count():reveal.first.click()
  teach=page.locator('#teachback200');area=teach.locator('textarea');assert area.count()==1
  area.fill('A complete inference run cannot establish my hypothesis or defend its controls. The proposal and written defense are independent learner evidence.')
  for button in teach.locator('button').all():
   if button.is_enabled():button.click();break
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),f'Page overflow {width}'
  assert page.locator('.exit-figure img').evaluate('(el)=>el.complete&&el.naturalWidth>0')
  page.evaluate('window.scrollTo(0,0)');page.screenshot(path=str(V/f'top-{width}.png'))
  board.screenshot(path=str(V/f'gates-{width}.png'));page.locator('.exit-figure').screenshot(path=str(V/f'architecture-{width}.png'))
 page.emulate_media(media='print');assert board.locator('.exit-controls').evaluate('(el)=>getComputedStyle(el).display')=='none';page.screenshot(path=str(V/'print.png'))
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri())
 assert 'Baseline: reproduction PASS' in pg.locator('body').inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.set_viewport_size({'width':1200,'height':950});page.goto((P/'html'/(S+'.html')).as_uri())
 assert 'COMPLETE_SELECTED_REPRODUCTION' in page.locator('body').inner_text()
 img=page.locator('img[alt="RDB-PFN information flow"]');assert img.count()==1 and img.evaluate('(el)=>el.complete && el.naturalWidth>0');img.screenshot(path=str(V/'notebook-figure.png'))
 assert not errors,errors;browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
links=0
for name in ['lessons/'+S+'.html','reference/year-5-exit-exam.html']:
 path=R/name;parser=Links();parser.feed(path.read_text());assert '[[' not in path.read_text()
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
manifest=json.loads((R/'lessons/manifest.json').read_text());assert sum(x['id']==200 for x in manifest['lessons'])==1
for name in ['index.html','notebooks.html']:assert 'manifest' in (R/name).read_text()
result=dict(status='PASS',browser_states=states,widths=[1200,375],keyboard_reset='PASS',no_javascript_print='PASS',local_links=links,visible_source_parity='EXACT',standalone_zip='PASS',deterministic_build='PASS',manifest_navigation='PASS',console_errors=errors,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l200_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
