"""Browser interaction and delivery checks; reports only observed results."""
import ast,hashlib,json,os,re,subprocess,sys,tempfile,threading,functools
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0144-contextgnn'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert all(not c.outputs for c in student.cells if c.cell_type=='code')
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');digest=hashlib.sha256(code.encode()).hexdigest()
assert 'from relkit' not in code and 'import relkit' not in code
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert digest==json.loads((P/'_execution_l144_results.json').read_text())['executed_code_sha256']
inline={n.name:n for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for file in ['relkit/context_l144.py','relkit/contextgnn_l144.py','_full_l144.py']:
 for n in ast.parse((P/file).read_text()).body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)):assert ast.dump(n,include_attributes=False)==ast.dump(inline[n.name],include_attributes=False),(file,n.name)
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('requestfailed',lambda r:errors.append(r.url))
 for width in [1200,375]:
  page.set_viewport_size(dict(width=width,height=900));page.goto((R/'lessons'/f'{S}.html').as_uri())
  host=page.locator('#context-ranking');slider=host.locator('input[type=range]');check=host.locator('input[type=checkbox]')
  for member in [True,False]:
   check.set_checked(member)
   for offset in [-3,-1,0,1.5,3]:
    slider.fill(str(offset));slider.dispatch_event('input');scores=[3+offset,2,1+offset if member else 3,4];order=sorted(range(4),key=lambda i:(-scores[i],i));ranking=' → '.join('ABCD'[i] for i in order)
    assert 'Ranking: '+ranking+'.' in host.inner_text();states+=1
  host.locator('button').click();assert slider.input_value()=='0' and check.is_checked();slider.focus();page.keyboard.press('ArrowRight');assert slider.input_value()=='.5' or slider.input_value()=='0.5';host.locator('button').click()
  assert page.locator('#warmup button').count()==0 and page.locator('#context-teachback textarea').count()==1
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===4&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
  page.screenshot(path=f'/tmp/l144-top-{width}.png');host.screenshot(path=f'/tmp/l144-ranking-{width}.png')
  for i in range(4):page.locator('figure').nth(i).screenshot(path=f'/tmp/l144-figure-{i}-{width}.png')
 page.emulate_media(media='print');page.pdf(path='/tmp/l144-print.pdf',format='A4');assert Path('/tmp/l144-print.pdf').stat().st_size>20000
 nojs=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=nojs.new_page();pg.goto((R/'lessons'/f'{S}.html').as_uri());assert 'Baseline:' in pg.locator('noscript').inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');nojs.close();browser.close()
assert not errors,errors
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0];lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))];checked=0
with tempfile.TemporaryDirectory(prefix='l144-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/contextgnn.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/contextgnn_l144.py','labs/l144-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l144/summary.json','labs/_full_l144.py','modal/l144_repro.py','labs/sources/l144/contextgnn/nn/models/contextgnn.py']:assert (stage/rel).is_file(),rel
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start()
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
  page.goto(f'http://127.0.0.1:{server.server_port}/index.html');assert reveal_gallery_link(page,'a[href="lessons/'+S+'.html"]').is_visible()
  page.goto(f'http://127.0.0.1:{server.server_port}/notebooks.html');assert reveal_gallery_link(page,'a[href="labs/html/'+S+'.html"]').is_visible();browser.close()
finally:server.shutdown();server.server_close()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths=[R/'lessons'/f'{S}.html',R/'reference/contextgnn.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb']+list((P/'figures/l144').glob('*'));before={str(p):sha(p) for p in paths}
subprocess.run([sys.executable,str(P/'_figures_l144.py')],cwd=R,check=True,capture_output=True);subprocess.run([sys.executable,str(P/'_build_l144.py')],cwd=R,check=True,capture_output=True)
subprocess.run([sys.executable,str(R/'scripts/refresh_lesson_visuals.py')],cwd=R,check=True,capture_output=True)
assert before=={str(p):sha(p) for p in paths},'Nondeterministic build'
report=dict(status='PASS',browser_states=states,widths=[1200,375],keyboard_reset=True,no_javascript=True,print=True,portable_figures=4,copied_pages_links=checked,deterministic_build=True,console_errors=errors,manifest_galleries=True,canonical_inline_ast='PASS',notebook_code_sha256=digest,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l144_results.json').write_text(json.dumps(report,indent=2));print(report)
