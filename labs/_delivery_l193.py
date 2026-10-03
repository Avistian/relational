"""Browser, links, inline implementation and deterministic-builder verification."""
import ast,functools,hashlib,itertools,json,os,subprocess,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
from relkit.multitask_l193 import aggregate_suite
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0193-open-fm-full-task-set';E=P/'evidence/l193'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
solcode='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');stucode='\n\n'.join(c.source for c in student.cells if c.cell_type=='code')
assert stucode.count('NotImplementedError')==3 and 'NotImplementedError' not in solcode
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
soldefs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(solcode).body if isinstance(n,ast.FunctionDef)}
for path in [P/'relkit/multitask_l193.py',P/'_replay_l193.py']:
 for n in ast.parse(path.read_text()).body:
  if isinstance(n,ast.FunctionDef):assert soldefs[n.name]==ast.dump(n,include_attributes=False),n.name
assert json.loads((P/'_execution_l193_results.json').read_text())['executed_code_sha256']==hashlib.sha256(solcode.encode()).hexdigest()
errors=[];states=0;probe=json.loads((E/'packet/preprocessing.json').read_text())
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto((R/'lessons'/(S+'.html')).as_uri());host=page.locator('#coverage-explorer')
  for bits in itertools.product([False,True],repeat=8):
   host.evaluate('(host,bits)=>{host.querySelectorAll("input").forEach((x,i)=>x.checked=bits[i]);host.dispatchEvent(new Event("change",{bubbles:true}));}',list(bits))
   tasks=[dict(id=str(i),group='classification',metric='AUROC') for i in range(8)]
   summaries=[dict(task=str(i),status='COMPLETE' if b else 'INCOMPLETE',mean=([.9]*3+[.5]*5)[i] if b else None) for i,b in enumerate(bits)]
   expected=aggregate_suite(tasks,summaries)['classification']['mean'];got=json.loads(host.get_attribute('data-mean'))
   assert (expected is None and got is None) or abs(expected-got)<1e-14
   assert int(host.get_attribute('data-count'))==sum(bits);states+=1
  host.locator('[data-reset]').click();assert host.get_attribute('data-count')=='3'
  host.locator('input').first.focus();page.keyboard.press('Space');assert host.get_attribute('data-count')=='2';host.locator('[data-reset]').click()
  cat=page.locator('#category-explorer')
  for obs in probe['observations']:
   cat.locator('select').select_option(obs['unseen']);assert int(cat.get_attribute('data-changed'))==obs['changed_codes'];states+=1
  cat.locator('[data-reset]').click();assert cat.locator('select').input_value()=='a'
  quiz=page.locator('#multitask-quiz');assert len(set(len(s.split()) for s in quiz.locator('button').all_inner_texts()))==1
  quiz.locator('button').first.click();assert 'correct' in quiz.locator('.quiz-feedback').get_attribute('class')
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===3&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),f'Overflow {width}'
  host.screenshot(path=f'/tmp/l193-coverage-{width}.png');cat.screenshot(path=f'/tmp/l193-category-{width}.png');page.locator('figure').first.screenshot(path=f'/tmp/l193-architecture-{width}.png');page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l193-top-{width}.png')
 page.emulate_media(media='print');assert host.locator('.controls').evaluate('(x)=>getComputedStyle(x).display')=='none'
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert '0 / 630' in pg.locator('body').inner_text();assert pg.locator('noscript').count()==3;assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==3;page.locator('img[src^="data:image/png"]').first.screenshot(path='/tmp/l193-notebook.png')
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(SimpleHTTPRequestHandler,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start()
 try:
  for name in ['index.html','notebooks.html']:
   page.goto(f'http://127.0.0.1:{server.server_port}/'+name);reveal_gallery_link(page,'a[href*="'+S+'"]')
 finally:server.shutdown()
 assert not errors,errors;browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
links=0
for name in ['lessons/'+S+'.html','reference/open-fm-full-task-set.html']:
 path=R/name;parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
outputs=[R/'lessons'/(S+'.html'),R/'reference/open-fm-full-task-set.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb')]+list((P/'figures/l193').glob('*'))
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs};subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l193.py')],check=True)
assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},'Nondeterministic builder'
result=dict(status='PASS',browser_states=states,widths=[1200,375],keyboard_reset=True,no_js=True,print=True,inline_source_parity=True,deterministic_build=True,local_links=links,galleries=True,portable_figures=3,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l193_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
