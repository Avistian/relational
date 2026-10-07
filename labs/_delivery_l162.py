"""Browser, copied-site, source-parity and deterministic-build checks."""
import signal
signal.alarm(600)
import ast,functools,hashlib,json,os,re,subprocess,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0162-the-relational-fm-vision'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert all(not c.outputs for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert sum(c.source.count('data:image/png;base64,') for c in solution.cells)==3
assert not any('def reachable_rows' in c.source or 'def token_budget' in c.source or 'def evidence_verdict' in c.source for c in student.cells if c.cell_type=='code' and 'raise NotImplementedError' not in c.source)
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_l162_results.json').read_text())['executed_code_sha256']
# All inherited computations must also appear verbatim as readable function cells.
function_cells={ast.dump(n,include_attributes=False) for c in solution.cells if c.cell_type=='code' for n in ast.parse(c.source).body if isinstance(n,ast.FunctionDef)}
for file in ['relkit/vision_l162.py','_run_l162.py','_check_l162.py']:
 for node in ast.parse((P/file).read_text()).body:
  if isinstance(node,ast.FunctionDef):assert ast.dump(node,include_attributes=False) in function_cells,(file,node.name)
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('requestfailed',lambda req:errors.append(req.url))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/(S+'.html')).as_uri())
  host=page.locator('#context-explorer')
  for hops in range(4):
   for edge in [False,True]:
    for planet in [False,True]:
     host.locator('[data-hops]').select_option(str(hops));host.locator('[data-edge]').set_checked(edge);host.locator('[data-planet]').set_checked(planet)
     # Use the independent canonical Python calculation, not duplicated JS logic.
     from relkit.vision_l162 import reachable_rows,token_budget
     nodes={'m1':'moons','m2':'moons','p1':'planets','s1':'stars','z1':'isolated'}
     edges=[['m1','p1'],['m2','p1']]+([['p1','s1']] if edge else [])
     expected=reachable_rows(nodes,edges,'m1',hops,None if planet else ['moons','stars','isolated'])
     assert host.locator('output').inner_text().startswith(', '.join(expected)+' · '+str(len(expected))+' reachable')
     assert 'Fixed baseline' in host.inner_text();states+=1
  host.locator('[data-reset]').click();assert host.locator('[data-hops]').input_value()=='2'
  box=host.locator('[data-planet]');box.focus();page.keyboard.press('Space');assert not box.is_checked();assert host.locator('output').inner_text().startswith('m1 · 1 reachable')
  host.locator('[data-reset]').click();assert box.is_checked()
  boundary=page.locator('#token-explorer')
  for kind,lengths in [('two',[3,5]),('many',[8]*100),('wide',[1100,18,22])]:
   for limit in [4,8,1024]:
    boundary.locator('[data-lengths]').select_option(kind);boundary.locator('[data-limit]').select_option(str(limit))
    expected=token_budget(lengths,limit)
    for selector,key in [('data-whole','whole_table_pairs'),('data-rows','row_pairs'),('data-kept','retained_row_pairs')]:assert boundary.locator('['+selector+']').inner_text()==str(expected[key])+' pairs'
    assert boundary.locator('output').inner_text()==str(len(lengths))+' rows · '+str(len(expected['overflow_rows']))+' over limit · '+str(expected['dropped_tokens'])+' dropped tokens'
    states+=1
  boundary.locator('[data-reset]').click();assert boundary.locator('[data-lengths]').input_value()=='two' and boundary.locator('[data-limit]').input_value()=='4'
  assert page.locator('#warmup button').count()==0
  predict=page.locator('#predict');assert predict.locator('.predict-reveal').is_disabled()
  predict.locator('[data-value="no"]').click();predict.locator('.predict-reveal').click();assert 'Removing p1' in predict.locator('.predict-outcome').inner_text()
  teach=page.locator('#teachback');assert teach.locator('textarea').count()==1
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===2&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
  page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l162-top-{width}.png')
  host.screenshot(path=f'/tmp/l162-adaptation-{width}.png');boundary.screenshot(path=f'/tmp/l162-boundary-{width}.png')
  for i in range(2):page.locator('figure').nth(i).screenshot(path=f'/tmp/l162-figure-{i}-{width}.png')
 page.emulate_media(media='print');assert page.locator('.fm-controls').first.evaluate('(x)=>getComputedStyle(x).display')=='none'
 page.screenshot(path='/tmp/l162-print.png',full_page=True)
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert pg.locator('noscript').count()==2;assert '16 graph interventions' in pg.locator('body').inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.set_viewport_size({'width':1000,'height':900});page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==3
 assert 'TRANSFER_REVIEW_ELIGIBLE' in page.locator('body').inner_text()
 page.locator('img[src^="data:image/png"]').first.screenshot(path='/tmp/l162-notebook-figure.png')
 browser.close()
assert not errors,errors
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start()
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 for path,selector in [('index.html','a[href="lessons/'+S+'.html"]'),('notebooks.html','a[href="labs/html/'+S+'.html"]')]:
  page.goto('http://127.0.0.1:'+str(server.server_port)+'/'+path);reveal_gallery_link(page,selector)
 browser.close()
server.shutdown();server.server_close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[l[10:] for l in block.splitlines() if l.startswith('          ')];lines=[l for l in lines if not l.startswith(('VER=','sed -i'))];count=0
with tempfile.TemporaryDirectory(prefix='l162-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),l) for l in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/(S+'.html'),stage/'reference/relational-fm-vision.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   dest=(path.parent/unquote(part.path)).resolve();assert dest.exists(),str(dest);count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/relational-fm-vision.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'evidence/l162/report.json',P/'evidence/l162/report.md']+sorted((P/'figures/l162').iterdir())
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l162.py')],check=True,capture_output=True)
subprocess.run(['python3',str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
after=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
assert before==after,'Builder changed: '+str([str(p) for p,a,b in zip(paths,before,after) if a!=b])
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',no_js='PASS',print='PASS',
       inline_source_parity='PASS',portable_figures=3,notebook_code_cells=sum(c.cell_type=='code' for c in solution.cells),
       copied_pages_links=count,deterministic_build='PASS',manifest_galleries='PASS',transfer_performance='NOT_ESTABLISHED',javascript_errors=errors,
       live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l162_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
