"""Check visible-source parity, portable CLI, exact widget states, layout and navigation."""
import ast,functools,hashlib,itertools,json,os,subprocess,sys,tempfile,threading,zipfile
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
from relkit.proposals_l198 import interval_decision
R=Path(__file__).resolve().parents[1];P=R/'labs';E=P/'evidence/l198';Q=E/'packet';S='0198-three-research-directions'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
sc='\n\n'.join(c.source for c in student.cells if c.cell_type=='code');code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert sc.count('NotImplementedError')==3 and 'NotImplementedError' not in code
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert json.loads((P/'_execution_l198_results.json').read_text())['executed_code_sha256']==hashlib.sha256(code.encode()).hexdigest()
functions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
paths=[P/'relkit/proposals_l198.py',P/'_audit_l198.py',P/'_test_l198.py',Q/'relkit/gaps_l189.py',Q/'_audit_l189.py',Q/'relkit/landscape_l197.py',Q/'_audit_l197.py',Q/'evidence/l197/packet/relkit/tracking_l191.py',Q/'evidence/l197/packet/relkit/stress_l195.py',Q/'evidence/l197/packet/_replay_l191.py',Q/'evidence/l197/packet/_replay_l195.py']
for path in paths:
 for n in ast.parse(path.read_text()).body:
  if isinstance(n,ast.FunctionDef) and n.name!='run198':assert functions[n.name]==ast.dump(n,include_attributes=False),n.name
fixed=[R/'lessons'/(S+'.html'),R/'reference/three-research-directions.html',P/(S+'.ipynb'),E/'reproducer.zip',E/'ranked-proposals.md']
before={p:p.read_bytes() for p in fixed};executed=(P/'solutions'/(S+'.ipynb')).read_bytes()
subprocess.run([sys.executable,str(P/'_build_l198.py')],check=True,capture_output=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
rebuilt=nbformat.read(P/'solutions'/(S+'.ipynb'),4);assert '\n\n'.join(c.source for c in rebuilt.cells if c.cell_type=='code')==code
(P/'solutions'/(S+'.ipynb')).write_bytes(executed)
assert all(p.read_bytes()==b for p,b in before.items()),'Nondeterministic builder'
with tempfile.TemporaryDirectory(prefix='l198-zip-') as tmp:
 with zipfile.ZipFile(E/'reproducer.zip') as z:z.extractall(tmp)
 root=Path(tmp);subprocess.run([sys.executable,str(root/'labs/_audit_l198.py')],cwd=root,check=True,capture_output=True)
 assert json.loads((root/'labs/evidence/l198/report.json').read_text())==json.loads((E/'report.json').read_text())
errors=[];states=0;cases=json.loads((E/'proposals.json').read_text())
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto((R/'lessons'/(S+'.html')).as_uri())
  host=page.locator('#proposal-priority')
  for weights in itertools.product([1,2,3],repeat=3):
   for impact in [3,4]:
    for j,v in enumerate([*weights,impact]):host.locator('#rp-'+str(j)).fill(str(v))
    got=json.loads(host.get_attribute('data-result'))
    scores={c['id']:(impact if c['id']=='temporal' else c['impact'])*sum(a*b for a,b in zip(c['feasibility'],weights))/sum(weights) for c in cases}
    assert got['leaders']==sorted(k for k,v in scores.items() if v==max(scores.values()))
    assert all(abs(x['score']-scores[x['id']])<1e-12 for x in got['scores']);states+=1
  host.locator('button').click();assert json.loads(host.get_attribute('data-result'))['impact']==4
  host.locator('#rp-3').focus();page.keyboard.press('ArrowLeft');assert json.loads(host.get_attribute('data-result'))['impact']==3
  host.locator('button').click()
  panel=page.locator('#interval-explorer');levels=[-.04,-.01,-.005,0,.005,.01,.025,.04]
  for mode in ['benefit','sensitivity']:
   panel.locator('select').select_option(mode)
   for low,high in itertools.combinations_with_replacement(levels,2):
    panel.locator('[data-low]').fill(str(low));panel.locator('[data-high]').fill(str(high))
    assert panel.get_attribute('data-state')==interval_decision(low,high,.01,mode);states+=1
  panel.locator('[data-low]').fill('.04');panel.locator('[data-high]').fill('-.04');assert panel.get_attribute('data-state')=='INVALID_INTERVAL'
  panel.locator('button').click();assert panel.get_attribute('data-state')=='INCONCLUSIVE'
  panel.locator('[data-low]').focus();page.keyboard.press('ArrowUp');assert panel.locator('[data-low]').input_value()=='0'
  panel.locator('button').click()
  for detail in page.locator('.proposal-card').all():
   detail.locator('summary').focus();page.keyboard.press('Enter');assert detail.get_attribute('open') is not None
  assert page.locator('.proposal-card').count()==3
  assert '72' in page.locator('.proposal-card').nth(1).inner_text()
  assert not page.evaluate('document.documentElement.scrollWidth > innerWidth+1'),f'Overflow at {width}'
  # Assert the dynamic SVG labels and objects stay within the viewBox.
  assert panel.locator('svg').evaluate('(svg)=>[...svg.querySelectorAll("text,circle,rect,line")].every(e=>{const b=e.getBBox();return b.x>=0&&b.y>=0&&b.x+b.width<=601&&b.y+b.height<=161})')
  page.evaluate('window.scrollTo(0,0)');page.screenshot(path=f'/tmp/l198-top-{width}.png')
  panel.screenshot(path=f'/tmp/l198-interval-{width}.png');host.screenshot(path=f'/tmp/l198-priority-{width}.png')
  for j,figure in enumerate(page.locator('figure').all()):figure.screenshot(path=f'/tmp/l198-figure-{j}-{width}.png')
 page.emulate_media(media='print');assert panel.locator('.pd-controls').evaluate('(e)=>getComputedStyle(e).display')=='none'
 page.pdf(path='/tmp/l198-print.pdf',format='A4',print_background=True)
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri())
 assert 'Default priority: temporal 18.67' in pg.locator('body').inner_text();assert not pg.evaluate('document.documentElement.scrollWidth > innerWidth+1');context.close()
 page.emulate_media(media='screen');page.set_viewport_size({'width':1000,'height':950});page.goto((P/'html'/(S+'.html')).as_uri())
 assert 'COMPLETE_SELECTED_PROPOSAL_AUDIT' in page.locator('body').inner_text()
 for name in ['temporal','composite','transfer']:
  img=page.locator('img[alt="L198 '+name+'"]');assert img.count()==1 and img.evaluate('(e)=>e.complete&&e.naturalWidth>0');img.screenshot(path='/tmp/l198-notebook-'+name+'.png')
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
for name in ['lessons/'+S+'.html','reference/three-research-directions.html']:
 path=R/name;parser=Links();parser.feed(path.read_text());assert '[[' not in path.read_text()
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
result=dict(status='PASS',interactive_states=states,viewports=[1200,375],keyboard_reset_print_nojs='PASS',source_ast_parity='PASS',deterministic_builder='PASS',standalone_zip_cli='EXACT_REPORT',local_links=links,manifest_galleries='PASS',portable_figures=3,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l198_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
