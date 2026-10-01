"""Browser behavior, notebook parity, staging links and deterministic source build."""
import ast,functools,hashlib,json,os,re,subprocess,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
P=Path(__file__).resolve().parent;R=P.parent;S='0157-open-source-contribution'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert all(not c.outputs for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_l157_results.json').read_text())['executed_code_sha256']
source=ast.parse((P/'relkit/contribution_l157.py').read_text());expected={n.name:ast.dump(n,include_attributes=False) for n in source.body if isinstance(n,ast.FunctionDef)}
for name,node in expected.items():
 cells=[c for c in solution.cells if c.cell_type=='code' and c.source.startswith('# TODO:') and 'def '+name+'(' in c.source];assert len(cells)==1
 actual=next(n for n in ast.parse(cells[0].source).body if isinstance(n,ast.FunctionDef));assert ast.dump(actual,include_attributes=False)==node
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/(S+'.html')).as_uri());host=page.locator('#l157-review')
  for integrity in ['PASS','FAIL']:
   host.locator('[data-integrity]').select_option(integrity)
   for coverage in ['COMPLETE','INCOMPLETE']:
    host.locator('[data-coverage]').select_option(coverage)
    for claim in ['selected_reproduction','historically_leak_free','public_contribution','whole_paper','upstream_bug']:
     host.locator('[data-claim]').select_option(claim)
     for url in ['none','provided']:
      host.locator('[data-url]').select_option(url)
      expected='REJECTED' if integrity=='FAIL' else ('SUPPORTED' if coverage=='COMPLETE' else 'INCOMPLETE') if claim=='selected_reproduction' else ('PENDING_PUBLICATION' if url=='none' else 'NOT_CHECKED') if claim=='public_contribution' else 'NOT_RUN' if claim=='whole_paper' else 'NOT_ESTABLISHED'
      assert host.get_attribute('data-status')==expected;states+=1
  host.locator('button').click();assert host.get_attribute('data-status')=='SUPPORTED'
  host.locator('[data-integrity]').focus();page.keyboard.press('ArrowDown');assert host.get_attribute('data-status')=='REJECTED';host.locator('button').click()
  assert page.locator('#warmup button').count()>0
  teach=page.locator('#teachback');assert teach.locator('textarea').count()==1
  assert teach.locator('button').first.is_disabled()
  teach.locator('textarea').fill('Replay checks captured evidence; a fresh run repeats the experiment. Publication establishes access, and neither proves historical feature availability.')
  teach.locator('button').first.click();assert 'separate' in teach.inner_text()
  predict=page.locator('#predict');assert predict.locator('.predict-reveal').is_disabled();predict.locator('[data-value="scoped"]').click();predict.locator('.predict-reveal').click();assert 'arrival' in predict.locator('.predict-outcome').inner_text()
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===2&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
  page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l157-top-{width}.png');host.screenshot(path=f'/tmp/l157-review-{width}.png')
  page.locator('figure').first.screenshot(path=f'/tmp/l157-flow-{width}.png')
 page.emulate_media(media='print');assert page.locator('article').is_visible();page.screenshot(path='/tmp/l157-print.png',full_page=True)
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert 'PENDING_PUBLICATION' in pg.locator('noscript').inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==2;browser.close()
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
with tempfile.TemporaryDirectory(prefix='l157-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),l) for l in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/(S+'.html'),stage/'reference/reproducibility-contribution.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   dest=(path.parent/unquote(part.path)).resolve();assert dest.exists(),str(dest);count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/reproducibility-contribution.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'releases/l157-f1-audit.zip']+sorted((P/'figures/l157').iterdir())
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
for builder in ['_release_l157.py','_figures_l157.py','_build_l157.py']:subprocess.run([str(R/'.venv/bin/python'),str(P/builder)],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'Non-deterministic builder'
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',no_js='PASS',print='PASS',inline_source_parity='PASS',portable_figures=2,copied_pages_links=count,deterministic_build='PASS',manifest_galleries='PASS',javascript_errors=errors,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l157_results.json').write_text(json.dumps(r,indent=2));print(r)
