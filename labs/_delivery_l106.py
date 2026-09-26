"""Actual artifacts, browser interaction, notebook portability and copied Pages staging."""
import functools,hashlib,json,os,re,subprocess,sys,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0106-temporal-link-prediction'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=json.loads((P/'_analysis_l106_results.json').read_text())
assert sha(P/'relkit/edgebank_l106.py')==report['implementation_sha256']
assert sha(P/'_run_l106.py')==report['runner_sha256']
for name,h in report['artifacts'].items():assert sha(P/'evidence/l106'/name)==h
student=nbformat.read(P/f'{S}.ipynb',as_version=4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert not any(c.outputs for c in student.cells if c.cell_type=='code')
source='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert hashlib.sha256(source.encode()).hexdigest()==json.loads((P/'_execution_l106_results.json').read_text())['executed_code_sha256']
for chunk in re.split(r'^# %% ',(P/'relkit/edgebank_l106.py').read_text(),flags=re.M)[1:]:assert chunk.split('\n',1)[1].strip() in source
paths=[R/'lessons'/f'{S}.html',R/'reference/temporal-link-evaluation.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb']
before=[sha(x) for x in paths];subprocess.run([sys.executable,str(P/'_build_l106.py')],check=True,capture_output=True);assert before==[sha(x) for x in paths]
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
 page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('requestfailed',lambda r:errors.append(r.url))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri());widget=page.locator('#l106-candidates');select=widget.locator('select')
  for k,ap,auc in [('0','0.7500','0.7500'),('1','0.5000','0.5000'),('2','0.4167','0.2500')]:
   select.select_option(k);assert widget.locator('.ap').inner_text()==ap;assert widget.locator('.auc').inner_text()==auc;assert widget.locator('.stream-scroll').evaluate('(e)=>e.scrollWidth<=e.clientWidth+1');states+=1
   widget.screenshot(path=f'/tmp/l106-widget-{width}-{k}.png')
  widget.locator('button').click();assert select.input_value()=='0';select.focus();page.keyboard.press('ArrowDown');assert select.input_value()=='1';widget.locator('button').click()
  pred=page.locator('#l106-predict');assert pred.locator('button').last.is_disabled();pred.locator('button').first.click();pred.locator('button').last.click();assert 'without changing the scorer' in pred.inner_text()
  teach=page.locator('#l106-teachback');teach.locator('textarea').fill('The score compares supplied positives with sampled negatives at supplied query times. Historical negatives test obsolete memories, not just unseen endpoint combinations. I need a declared history and forecast horizon, and I must distinguish batch means from pooled scores. Matching released predictions does not certify historical paper or deployment identity.');teach.locator('button').first.click();assert 'Sparse random pairs' in teach.inner_text()
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),'page overflow'
  page.screenshot(path=f'/tmp/l106-page-{width}.png',full_page=True)
  assert page.locator('figure').count()==4
 for name in ['architecture','candidates','metrics','results']:
  page.goto((P/f'figures/l106/{name}.svg').as_uri())
  assert page.locator('svg').evaluate("s=>{const r=s.getBoundingClientRect();return Array.from(s.querySelectorAll('text')).every(t=>{const b=t.getBoundingClientRect();return b.x>=r.x-1&&b.y>=r.y-1&&b.right<=r.right+1&&b.bottom<=r.bottom+1})}"),name+' text outside canvas'
 page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('img[src^="data:image/png;base64,"]').count()==4
 page.set_viewport_size({'width':950,'height':900});page.locator('img[src^="data:image/png;base64,"]').nth(2).screenshot(path='/tmp/l106-notebook-metrics.png')
 nojs=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});plain=nojs.new_page();plain.goto((R/'lessons'/f'{S}.html').as_uri())
 assert plain.locator('figure').count()==4 and 'Without opening your notes' in plain.locator('article').inner_text();assert plain.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
 plain.emulate_media(media='print');assert plain.locator('details p').first.evaluate('(e)=>e.checkVisibility()');nojs.close();browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
(P/'_delivery_l106_results.json').write_text('{"status":"RUNNING"}\n')
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))]
with tempfile.TemporaryDirectory(prefix='l106-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True);checked=0
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/temporal-link-evaluation.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/edgebank_l106.py','labs/_run_l106.py','labs/l106-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l106/split.npz','labs/sources/l106/edge_sampler.py']:
  assert (stage/rel).exists(),rel
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));base=f'http://127.0.0.1:{server.server_port}'
   page.goto(base+'/index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]');assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(base+'/notebooks.html');page.wait_for_selector('a[href="labs/html/'+S+'.html"]')
   page.goto(base+'/lessons/'+S+'.html');assert page.locator('#l106-candidates .ap').inner_text()=='0.7500';browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
out={'status':'PASS','browser_widths':[1200,375],'candidate_states':states,'reset_keyboard':'PASS','prediction_and_teachback':'PASS','print_nojs':'PASS','portable_figures':4,'student_live_tasks':3,'canonical_inline_source':'EXACT','executed_code_hash':'MATCH','evidence_hashes':'MATCH','copied_pages_links':checked,'deterministic_rebuild':'EXACT','manifest_navigation':'PASS','javascript_errors':errors,'screenshots':'/tmp/l106-*.png','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/'_delivery_l106_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
