"""Rendered desktop/mobile interaction, portable-source and copied-Pages checks."""
import inspect,ast,functools,hashlib,json,os,re,subprocess,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0155-compare-manual-fe'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert all(not c.outputs for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert sum(c.source.count('data:image/png;base64,') for c in solution.cells)==3
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_l155_results.json').read_text())['executed_code_sha256']
class NormalizeDocs(ast.NodeTransformer):
 def visit_FunctionDef(self,node):
  self.generic_visit(node)
  if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body[0].value.value=inspect.cleandoc(node.body[0].value.value)
  return node
 visit_ClassDef=visit_FunctionDef
normalize=NormalizeDocs()
inline={n.name:n for n in ast.walk(ast.parse(code)) if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for file,names in [('relkit/effort_l155.py',['paired_losses','summarize_effort','effort_ratio']),('_report_l155.py',['make_report','render_report']),('relkit/fe_experiment_l129.py',['load_archives','make_features','materialize_features','tune_fe']),('relkit/rdl_l117.py',['Model','fit_rdl','make_pkey_fkey_graph'])]:
 nodes={n.name:n for n in ast.parse((P/file).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 for name in names:
  expected=nodes[name]
  if name=='tune_fe':expected.body=[n for n in expected.body if not isinstance(n,ast.ImportFrom)]
  assert ast.dump(normalize.visit(expected),include_attributes=False)==ast.dump(normalize.visit(inline[name]),include_attributes=False),name
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/(S+'.html')).as_uri())
  host=page.locator('#l155-effort');scope=host.locator('[data-scope]');tasks=host.locator('[data-tasks]');observed=host.locator('[data-observed]')
  for s in ['marginal','amortized']:
   scope.select_option(s)
   for n in [1,10,20]:
    tasks.fill(str(n));tasks.dispatch_event('input')
    for known in [True,False]:
     observed.set_checked(known);expected=2/(.5+(1.5/n if s=='amortized' else 0))
     if known:assert abs(float(host.get_attribute('data-ratio'))-expected)<1e-12
     else:assert host.get_attribute('data-ratio')=='' and 'NOT_OBSERVED' in host.inner_text()
     states+=1
  host.locator('[data-reset]').click();assert scope.input_value()=='marginal' and tasks.input_value()=='1' and observed.is_checked()
  observed.focus();page.keyboard.press('Space');assert not observed.is_checked();host.locator('[data-reset]').click()
  tasks.focus();page.keyboard.press('ArrowRight');assert tasks.input_value()=='2';host.locator('[data-reset]').click()
  assert page.locator('#warmup button').count()>0
  predict=page.locator('#predict');assert predict.locator('.predict-reveal').is_disabled()
  predict.locator('[data-value="compare"]').click();predict.locator('.predict-reveal').click();assert 'matched query losses' in predict.locator('.predict-outcome').inner_text()
  assert page.locator('#teachback textarea').count()==1
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===3&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
  page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l155-top-{width}.png');host.screenshot(path=f'/tmp/l155-widget-{width}.png')
  for i in range(3):page.locator('figure').nth(i).screenshot(path=f'/tmp/l155-figure-{i}-{width}.png')
 page.emulate_media(media='print');page.pdf(path='/tmp/l155-print.pdf',format='A4');assert Path('/tmp/l155-print.pdf').stat().st_size>20000
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert pg.locator('noscript').count()==1;assert 'NOT_OBSERVED' in pg.locator('body').inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==3;assert '3.948917' in page.locator('body').inner_text();browser.close()
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
with tempfile.TemporaryDirectory(prefix='l155-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),l) for l in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/(S+'.html'),stage/'reference/compare-manual-fe.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   dest=(path.parent/unquote(part.path)).resolve();assert dest.exists(),str(dest);count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/compare-manual-fe.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'evidence/l155/report.json',P/'evidence/l155/report.md']+sorted((P/'figures/l155').iterdir())
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l155.py')],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'Builder is not deterministic'
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',no_js='PASS',print='PASS',inline_source_parity='PASS',portable_figures=3,notebook_code_cells=sum(c.cell_type=='code' for c in solution.cells),copied_pages_links=count,deterministic_build='PASS',manifest_galleries='PASS',javascript_errors=errors,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l155_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
