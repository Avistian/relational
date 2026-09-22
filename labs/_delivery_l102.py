"""Browser interaction, figure geometry, portable notebooks and copied Pages verification."""
import functools,hashlib,json,math,os,re,subprocess,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0102-temporal-graph-networks'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
errors=[];states=0
# Independent widget oracle: process complete preceding batches, then score selected event.
events=[('A','B',1),('A','C',2),('B','C',-1),('A','B',.5)]
def oracle(index,batch,leak):
 state=dict(A=0.,B=0.,C=0.)
 for start in range(0,(index//batch)*batch,batch):
  old=state.copy()
  for a,b,e in events[start:start+batch]:
   state[a]=.5*old[a]+.5*math.tanh(e+.25*old[a]);state[b]=.5*old[b]+.5*math.tanh(e+.25*old[b])
 a,b,e=events[index]
 if leak:
  state[a]=.5*state[a]+.5*math.tanh(e+.25*state[a]);state[b]=.5*state[b]+.5*math.tanh(e+.25*state[b])
 return 1/(1+math.exp(-state[a]-state[b]))
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
 page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri())
  widget=page.locator('#temporal-graph');slider=widget.locator('input[type=range]');select=widget.locator('select');checkbox=widget.locator('input[type=checkbox]');out=widget.locator('.temporal-readout')
  assert 'Legal score: 0.5000' in out.inner_text()
  for batch in [1,2]:
   select.select_option(str(batch))
   for leak in [False,True]:
    checkbox.set_checked(leak)
    for idx in range(4):
     slider.fill(str(idx));slider.dispatch_event('input')
     assert f'Legal score: {oracle(idx,batch,False):.4f}' in out.inner_text()
     assert f'Compared score: {oracle(idx,batch,leak):.4f}' in out.inner_text();states+=1
  widget.locator('button').click();slider.focus();page.keyboard.press('ArrowRight');assert slider.input_value()=='2';widget.locator('button').click()
  assert page.locator('#l102-warmup').inner_text().strip()
  pred=page.locator('#l102-predict');assert pred.locator('button').last.is_disabled();pred.locator('button').first.click();pred.locator('button').last.click()
  teach=page.locator('#l102-teachback');teach.locator('textarea').fill('Past observed messages enter a new differentiable GRU update before scoring the current event. Only afterward does that event become a pending message. Weights freeze during evaluation, but memory keeps evolving with observed history.');teach.locator('button').first.click()
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'page overflow'
  for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
  page.evaluate('window.scrollTo(0,0)');page.screenshot(path=f'/tmp/l102-top-{width}.png');widget.screenshot(path=f'/tmp/l102-widget-{width}.png');page.locator('figure').first.screenshot(path=f'/tmp/l102-architecture-{width}.png')
 page.emulate_media(media='print');assert page.locator('h1').is_visible();page.emulate_media(media='screen')
 for path in (P/'figures/l102').glob('*.svg'):
  page.goto(path.as_uri())
  if path.stem!='results':
   assert page.locator('svg').evaluate("s=>{let v=s.viewBox.baseVal;return Array.from(s.querySelectorAll('text')).every(t=>{let b=t.getBBox();return b.x>=0&&b.y>=0&&b.x+b.width<=v.width&&b.y+b.height<=v.height})}"),str(path)+' text overflow'
   assert page.locator('svg').evaluate("s=>{let v=s.viewBox.baseVal;let b=Array.from(s.querySelectorAll('rect[rx]')).map(e=>e.getBBox());return b.every((x,i)=>x.x>=0&&x.y>=0&&x.x+x.width<=v.width&&x.y+x.height<=v.height&&b.every((y,j)=>i===j||x.x+x.width<=y.x||y.x+y.width<=x.x||x.y+x.height<=y.y||y.y+y.height<=x.y))}"),'overlapping boxes'
 page.goto((P/'html'/f'{S}.html').as_uri());figure_count=page.locator('img[src^="data:image/png;base64,"]').count();assert figure_count>=3
 browser.close()
student=nbformat.read(P/f'{S}.ipynb',as_version=4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
assert sum('raise NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==3
assert all(not c.outputs for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert 'from relkit' not in '\n'.join(c.source for c in solution.cells)
paths=[R/'lessons'/f'{S}.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb']
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l102.py')],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'nondeterministic build'
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
(P/'_delivery_l102_results.json').write_text('{"status":"RUNNING"}\n')
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0];lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))]
with tempfile.TemporaryDirectory(prefix='l102-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 checked=0
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/tgn-memory.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists(),(path,url);assert not dest.is_symlink();checked+=1
 for rel in ['labs/relkit/tgn_l102.py','labs/_verify_l102.py','labs/_plot_l102_results.py','labs/_record_l102_evidence.py','labs/sources/l102/model/tgn.py','labs/solutions/'+S+'.ipynb','modal/l102_paper_repro.py']:
  assert (stage/rel).exists(),rel
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page()
   page.goto(f'http://127.0.0.1:{server.server_port}/index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]')
   assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(f'http://127.0.0.1:{server.server_port}/lessons/{S}.html');assert 'Legal score: 0.5000' in page.locator('.temporal-readout').inner_text();browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
report={'status':'PASS','browser_widths':[1200,375],'widget_states':states,'independent_widget_arithmetic':'PASS','reset_and_keyboard':'PASS','print':'CHECKED','portable_figures':figure_count,'student_live_tasks':3,'copied_pages_local_links':checked,'deterministic_rebuild':'EXACT','manifest_navigation_over_http':'PASS','javascript_errors':errors,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/'_delivery_l102_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
