"""Actual browser, standalone-source, copied-site and deterministic-build checks."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import ast,functools,hashlib,json,math,re,subprocess,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0164-griffin-graph-centric-rdb-fm'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert all(not c.outputs for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert sum(c.source.count('data:image/png;base64,') for c in solution.cells)==3
assert all('def '+name not in c.source for name in ['cell_attention','relation_pool','eligible_edges'] for c in student.cells if c.cell_type=='code' and 'raise NotImplementedError' not in c.source)
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_l164_results.json').read_text())['executed_code_sha256']
inline={ast.dump(n,include_attributes=False) for c in solution.cells if c.cell_type=='code' for n in ast.parse(c.source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for f in ['relkit/griffin_l164.py','_check_l164.py','_lesson_run_l164.py']:
 for n in ast.parse((P/f).read_text()).body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)):assert ast.dump(n,include_attributes=False) in inline,(f,n.name)
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('requestfailed',lambda req:errors.append(req.url))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/(S+'.html')).as_uri())
  host=page.locator('#cell-explorer')
  for task in ['A','B']:
   for order in ['same','reverse']:
    for mask in ['keep','hide']:
     host.locator('[data-task]').select_option(task);host.locator('[data-order]').select_option(order);host.locator('[data-mask]').select_option(mask)
     weight=math.e/(math.e+1) if task=='A' else 1/(math.e+1)
     want=[2.,0.] if mask=='hide' else [2*weight,4*(1-weight)]
     got=json.loads(host.get_attribute('data-result'));assert max(abs(a-b) for a,b in zip(want,got))<1e-12
     assert host.locator('tbody tr').count()==2;states+=1
  host.locator('[data-reset]').click();host.locator('[data-task]').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert host.locator('[data-task]').input_value()=='B';host.locator('[data-reset]').click()
  host.screenshot(path=f'/tmp/l164-cells-{width}.png')
  host=page.locator('#relation-explorer')
  for repeat in [1,2,4]:
   for cutoff in [5,7]:
    host.locator('[data-repeat]').select_option(str(repeat));host.locator('[data-cutoff]').select_option(str(cutoff))
    got=json.loads(host.get_attribute('data-result'));want=[4.,3.] if cutoff==5 else [7.,11.]
    assert got['out']==want
    n=repeat*2+(1 if cutoff==5 else 2)
    # Independent weighted neighbour sum: results sum[6,6] per repeat.
    totals=[6*repeat+4,6*repeat+2] if cutoff==5 else [6*repeat+14,6*repeat+22]
    assert max(abs(a-b/n) for a,b in zip(got['flat'],totals))<1e-12
    assert got['count']==n;states+=1
  host.locator('[data-reset]').click();assert host.locator('[data-repeat]').input_value()=='1';assert host.locator('[data-cutoff]').input_value()=='5'
  host.screenshot(path=f'/tmp/l164-relations-{width}.png')
  assert page.locator('#warmup button').count()>0
  predict=page.locator('#predict');assert predict.locator('.predict-reveal').is_disabled();predict.locator('[data-value="same"]').click();predict.locator('.predict-reveal').click();assert 'both double' in predict.locator('.predict-outcome').inner_text()
  assert page.locator('#teachback textarea').count()==1
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===3&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
  page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l164-top-{width}.png')
  for i in range(3):page.locator('figure').nth(i).screenshot(path=f'/tmp/l164-figure-{i}-{width}.png')
 page.emulate_media(media='print');assert page.locator('.griffin-controls').first.evaluate('(x)=>getComputedStyle(x).display')=='none'
 page.screenshot(path='/tmp/l164-print.png',full_page=True)
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert pg.locator('noscript').count()==1;assert 'INCOMPLETE_BUDGET_GATE' in pg.locator('body').inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.set_viewport_size({'width':1000,'height':900});page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==3
 assert 'SYNTHETIC_MECHANISM_AND_CACHED_REAL_ATTENTION' in page.locator('body').inner_text()
 page.locator('img[src^="data:image/png"]').first.screenshot(path='/tmp/l164-notebook-figure.png')
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
with tempfile.TemporaryDirectory(prefix='l164-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),l) for l in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/(S+'.html'),stage/'reference/griffin.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   dest=(path.parent/unquote(part.path)).resolve();assert dest.exists(),str(dest);count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/griffin.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'evidence/l164/report.json']+sorted((P/'figures/l164').iterdir())
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l164.py')],check=True,capture_output=True)
after=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
assert before==after,'Builder changed: '+str([str(p) for p,a,b in zip(paths,before,after) if a!=b])
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',no_js='PASS',print='PASS',inline_source_parity='PASS',portable_figures=3,notebook_code_cells=sum(c.cell_type=='code' for c in solution.cells),copied_pages_links=count,deterministic_build='PASS',manifest_galleries='PASS',javascript_errors=errors,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l164_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
