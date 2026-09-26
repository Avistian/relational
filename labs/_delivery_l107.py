"""Actual browser, notebook, links, copied Pages and deterministic build checks."""
import ast,functools,hashlib,json,os,re,subprocess,tempfile,threading,sys
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0107-snapshot-methods'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
student=nbformat.read(P/f'{S}.ipynb',as_version=4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert not any(c.outputs for c in student.cells if c.cell_type=='code')
source='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert hashlib.sha256(source.encode()).hexdigest()==json.loads((P/'_execution_l107_results.json').read_text())['executed_code_sha256']
parsed=ast.parse(source);definitions={n.name:ast.dump(n,include_attributes=False) for n in parsed.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
count=0
for name in ['snapshot_l107.py','sbm_l107.py','wiki_snapshot_l107.py','tgn_l102.py','auth_l107.py']:
 for node in ast.parse((P/'relkit'/name).read_text()).body:
  if isinstance(node,(ast.FunctionDef,ast.ClassDef)):assert definitions[node.name]==ast.dump(node,include_attributes=False),node.name;count+=1
figure_paths=sorted((P/'figures/l107').glob('*'))
figure_hashes=[sha(x) for x in figure_paths]
subprocess.run([sys.executable,str(P/'_figures_l107.py')],check=True,capture_output=True)
assert figure_hashes==[sha(x) for x in figure_paths],'Figure rebuild drift'
paths=[R/'lessons'/f'{S}.html',R/'reference/snapshot-state-contracts.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb']
before=[sha(x) for x in paths];subprocess.run([sys.executable,str(P/'_build_l107.py')],check=True,capture_output=True);assert before==[sha(x) for x in paths],'Rebuild drift'
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
 page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('requestfailed',lambda r:errors.append(r.url))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri())
  for mode in ['node','weight']:
   widget=page.locator('#l107-'+mode);gate=widget.locator('.gate');candidate=widget.locator('.candidate')
   for g in [0,.25,1]:
    for c in [-1,.8,1]:
     gate.fill(str(g));candidate.fill(str(c));gate.dispatch_event('input');candidate.dispatch_event('input')
     state=.2;expected=[]
     for x in [c,.1,.6]:state=g*state+(1-g)*x if mode=='node' else (1-g)*state+g*x;expected.append(f'{state:.4f}')
     assert widget.locator('.state').all_text_contents()==expected;states+=1
   widget.locator('button').click();assert gate.input_value()=='0.25';assert candidate.input_value()=='0.8'
   gate.focus();page.keyboard.press('ArrowRight');assert gate.input_value()=='0.3';widget.locator('button').click()
   widget.screenshot(path=f'/tmp/l107-widget-{mode}-{width}.png')
  pred=page.locator('#l107-predict');assert pred.locator('button').last.is_disabled();pred.locator('button').first.click();pred.locator('button').last.click();assert 'observation-free' in pred.inner_text()
  teach=page.locator('#l107-teachback');teach.locator('textarea').fill('At 09:05 I can use a graph closed at 09:00. The GCN computes spatial features; the recurrent state either has rows indexed by node identity or coordinates indexed by feature dimensions. Candidate scoring precedes updating with new observations. I would fix model weights and candidates while changing the historical access cutoff. A source replay differs from historical reproduction when recurrence and runtime are different.');teach.locator('button').first.click();assert 'snapshot' in teach.inner_text()
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),'page overflow'
  assert page.locator('figure').count()==6
  page.screenshot(path=f'/tmp/l107-page-{width}.png',full_page=True)
 for name in ['architecture','normalization','summary','recurrence','history','results']:
  page.goto((P/f'figures/l107/{name}.svg').as_uri());assert page.locator('svg').evaluate("s=>{const r=s.getBoundingClientRect();return Array.from(s.querySelectorAll('text')).every(t=>{const b=t.getBoundingClientRect();return b.x>=r.x-1&&b.y>=r.y-1&&b.right<=r.right+1&&b.bottom<=r.bottom+1})}"),name+' labels outside canvas'
 page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('img[src^="data:image/png;base64,"]').count()==6
 page.set_viewport_size({'width':950,'height':900});page.locator('img[src^="data:image/png;base64,"]').nth(2).screenshot(path='/tmp/l107-notebook-summary.png')
 nojs=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});plain=nojs.new_page();plain.goto((R/'lessons'/f'{S}.html').as_uri());assert plain.locator('figure').count()==6;assert 'First retrieve' in plain.locator('article').inner_text();assert plain.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
 plain.emulate_media(media='print');assert plain.locator('details p').first.evaluate('(e)=>e.checkVisibility()');nojs.close();browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
(P/'_delivery_l107_results.json').write_text('{"status":"RUNNING"}\n')
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))]
with tempfile.TemporaryDirectory(prefix='l107-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True);checked=0
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/snapshot-state-contracts.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/snapshot_l107.py','labs/relkit/sbm_l107.py','labs/relkit/wiki_snapshot_l107.py','labs/relkit/auth_l107.py','labs/_run_sbm_l107.py','labs/l107-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l107/wiki/tgn/seed-0/result.json','labs/sources/l107/original/egcn_h.py','modal/l107_repro.py']:
  assert (stage/rel).exists(),rel
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));base=f'http://127.0.0.1:{server.server_port}'
   page.goto(base+'/index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]');assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(base+'/notebooks.html');page.wait_for_selector('a[href="labs/html/'+S+'.html"]')
   page.goto(base+'/lessons/'+S+'.html');assert page.locator('#l107-weight .state').first.inner_text()=='0.3500';browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
out={'status':'PASS','browser_widths':[1200,375],'gate_candidate_states':states,'reset_keyboard':'PASS','prediction_and_teachback':'PASS','print_nojs':'PASS','portable_figures':6,'student_live_tasks':3,'canonical_definitions_exact':count,'executed_code_hash':'MATCH','copied_pages_links':checked,'deterministic_rebuild':'EXACT','deterministic_figures':'EXACT','manifest_navigation':'PASS','javascript_errors':errors,'screenshots':'/tmp/l107-*.png','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/'_delivery_l107_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
