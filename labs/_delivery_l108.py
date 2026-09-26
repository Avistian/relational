"""Validate actual lesson/notebook, interactive states and copied Pages artifacts."""
import ast,functools,hashlib,json,os,re,subprocess,tempfile,threading,sys
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0108-temporal-neighbor-sampling'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
student=nbformat.read(P/f'{S}.ipynb',as_version=4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert not any(c.outputs for c in student.cells if c.cell_type=='code')
source='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert hashlib.sha256(source.encode()).hexdigest()==json.loads((P/'_execution_l108_results.json').read_text())['executed_code_sha256']
definitions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
count=0
for name in ['sampling_l108.py','tgat_l103.py']:
 for node in ast.parse((P/'relkit'/name).read_text()).body:
  if isinstance(node,(ast.FunctionDef,ast.ClassDef)):assert definitions[node.name]==ast.dump(node,include_attributes=False),node.name;count+=1
r=json.loads((P/'_analysis_l108_results.json').read_text());assert r['status']=='COMPLETE'
for name,h in r['artifacts'].items():assert sha(P/'evidence/l108'/name)==h,name
paths=[R/'lessons'/f'{S}.html',R/'reference/temporal-sampling.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb']
before=[sha(x) for x in paths];subprocess.run([sys.executable,str(P/'_build_l108.py')],check=True,capture_output=True);assert before==[sha(x) for x in paths],'Rebuild drift'
figures=sorted((P/'figures/l108').glob('*'));before=[sha(x) for x in figures];subprocess.run([sys.executable,str(P/'_figures_l108.py')],check=True,capture_output=True);assert before==[sha(x) for x in figures],'Figure drift'
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
 page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('requestfailed',lambda r:errors.append(r.url))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri());widget=page.locator('#l108-sampler');cut=widget.locator('.cut');win=widget.locator('.win');fan=widget.locator('.fan')
  assert 'Sample times: [3, 3]' in widget.inner_text()
  for c in [1,8,9,13]:
   for w in [1,5,12]:
    for k in [1,2,3]:
     cut.fill(str(c));win.fill(str(w));fan.select_option(str(k));cut.dispatch_event('input');win.dispatch_event('input');fan.dispatch_event('input')
     eligible=[t for t in [1,3,3,8,11] if c-w<=t<c];chosen=eligible[-k:]
     assert 'Sample times: ['+', '.join(map(str,chosen))+']' in widget.locator('.answer').inner_text();states+=1
  widget.locator('button').click();assert cut.input_value()=='8' and win.input_value()=='5' and fan.input_value()=='2'
  cut.focus();page.keyboard.press('ArrowRight');assert 'Sample times: [8]' in widget.inner_text();widget.locator('button').click()
  widget.screenshot(path=f'/tmp/l108-widget-{width}.png')
  pred=page.locator('#l108-predict');assert pred.locator('button').last.is_disabled();pred.locator('button').first.click();pred.locator('button').last.click();assert 'same score' in pred.inner_text()
  teach=page.locator('#l108-teachback');teach.locator('textarea').fill('For cutoff eight and width five I use the half-open interval from three to eight. A selected edge at three passes cutoff three into the next hop. Two searches can preserve exact records; changing fanout changes the context. I compare the same questions and negative candidates, synchronize GPU timing, and distinguish fixed-weight inference from fresh training.');teach.locator('button').first.click();assert 'frozen model' in teach.inner_text()
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),'Page overflow'
  assert page.locator('figure').count()==4
  page.screenshot(path=f'/tmp/l108-page-{width}.png',full_page=True)
 for name in ['interval','pipeline','recursion','results']:
  page.goto((P/f'figures/l108/{name}.svg').as_uri());assert page.locator('svg').evaluate("s=>{const r=s.getBoundingClientRect();return Array.from(s.querySelectorAll('text')).every(t=>{const b=t.getBoundingClientRect();return b.x>=r.x-1&&b.y>=r.y-1&&b.right<=r.right+1&&b.bottom<=r.bottom+1})}"),name+' labels outside canvas'
 page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('img[src^="data:image/png;base64,"]').count()==4
 page.set_viewport_size({'width':950,'height':900});page.locator('img[src^="data:image/png;base64,"]').nth(0).screenshot(path='/tmp/l108-notebook-interval.png')
 nojs=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});plain=nojs.new_page();plain.goto((R/'lessons'/f'{S}.html').as_uri());assert plain.locator('figure').count()==4 and 'Worked answer, including a no-JavaScript fallback' in plain.locator('article').inner_text();assert plain.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
 plain.emulate_media(media='print');assert plain.locator('details p').first.evaluate('(e)=>e.checkVisibility()');plain.screenshot(path='/tmp/l108-print.png',full_page=True);nojs.close();browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
(P/'_delivery_l108_results.json').write_text('{"status":"RUNNING"}\n')
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))]
with tempfile.TemporaryDirectory(prefix='l108-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True);checked=0
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/temporal-sampling.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/sampling_l108.py','labs/_run_l108.py','labs/l108-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l108/full/seed-9/predictions.npz','modal/l108_replay.py']:
  assert (stage/rel).exists(),rel
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));base=f'http://127.0.0.1:{server.server_port}'
   page.goto(base+'/index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]');assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(base+'/notebooks.html');page.wait_for_selector('a[href="labs/html/'+S+'.html"]')
   page.goto(base+'/lessons/'+S+'.html');assert 'Sample times: [3, 3]' in page.locator('#l108-sampler').inner_text();browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
out={'status':'PASS','browser_widths':[1200,375],'widget_states':states,'reset_keyboard':'PASS','prediction_and_teachback':'PASS','print_nojs':'PASS','portable_figures':4,'student_live_tasks':3,'canonical_definitions':count,'executed_code_hash':'MATCH','evidence_hashes':'MATCH','copied_pages_links':checked,'deterministic_rebuild':'EXACT','manifest_navigation':'PASS','javascript_errors':errors,'screenshots':'/tmp/l108-*.png','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/'_delivery_l108_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
