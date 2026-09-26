"""Check interactions, responsive reading, portable notebook and copied Pages navigation."""
import functools,hashlib,json,math,os,re,subprocess,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0103-tgat'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
 page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri())
  widget=page.locator('#l103-time');slider=widget.locator('input');out=widget.locator('.tgat-readout')
  for t in [4,4.5,6,8,10]:
   slider.fill(str(t));slider.dispatch_event('input')
   x,y=math.exp(math.cos(t-1)),math.exp(math.cos(t-3));value=(2*x+8*y)/(x+y)
   assert f'Current weighted value: {value:.4f}' in out.inner_text();states+=1
  widget.locator('button').click();slider.focus();page.keyboard.press('ArrowRight');assert slider.input_value()=='4.1';widget.locator('button').click()
  assert page.locator('#l103-warmup').inner_text().strip()
  pred=page.locator('#l103-predict');assert pred.locator('button').last.is_disabled();pred.locator('button').first.click();pred.locator('button').last.click()
  teach=page.locator('#l103-teachback');teach.locator('textarea').fill('TGAT recomputes embeddings from historical events. The child cutoff is the connecting event time. Cosine time features influence keys and values, while TGN maintains per-node state. A released Wikipedia replay is not full-paper reproduction.');teach.locator('button').first.click()
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'page overflow'
  for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
  page.evaluate('window.scrollTo(0,0)');page.screenshot(path=f'/tmp/l103-top-{width}.png');widget.screenshot(path=f'/tmp/l103-widget-{width}.png');page.locator('figure').nth(2).screenshot(path=f'/tmp/l103-architecture-{width}.png')
 page.emulate_media(media='print');assert page.locator('h1').is_visible();page.emulate_media(media='screen')
 for name in ['architecture','recursion']:
  page.goto((P/f'figures/l103/{name}.svg').as_uri())
  assert page.locator('svg').evaluate("s=>{let v=s.viewBox.baseVal;return Array.from(s.querySelectorAll('text')).every(t=>{let b=t.getBBox();return b.x>=0&&b.y>=0&&b.x+b.width<=v.width&&b.y+b.height<=v.height})}"),name+' text overflow'
  assert page.locator('svg').evaluate("s=>{let boxes=Array.from(s.querySelectorAll('rect[rx]')).map(e=>e.getBBox());return boxes.every((x,i)=>boxes.every((y,j)=>i===j||x.x+x.width<=y.x||y.x+y.width<=x.x||x.y+x.height<=y.y||y.y+y.height<=x.y))}"),'box overlap'
 page.goto((P/'html'/f'{S}.html').as_uri());figures=page.locator('img[src^="data:image/png;base64,"]').count();assert figures>=3
 page.set_viewport_size({'width':950,'height':900});page.locator('img[src^="data:image/png;base64,"]').nth(2).screenshot(path='/tmp/l103-notebook-architecture.png')
 static=browser.new_context(java_script_enabled=False)
 plain=static.new_page()
 for width in [1200,375]:
  plain.set_viewport_size({'width':width,'height':900});plain.goto((R/'lessons'/f'{S}.html').as_uri())
  assert plain.locator('h1').is_visible() and 'TGAT' in plain.locator('h1').inner_text()
  assert 'PENDING_WRITTEN_DEFENSE' in plain.locator('body').inner_text()
  assert '[[' not in plain.locator('body').inner_text()
  assert plain.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
 static.close()
 browser.close()
student=nbformat.read(P/f'{S}.ipynb',as_version=4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
assert sum('raise NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==3
assert all(not c.outputs for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert 'from relkit' not in '\n'.join(c.source for c in solution.cells)
paths=[R/'lessons'/f'{S}.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb']
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l103.py')],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'nondeterministic build'
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
(P/'_delivery_l103_results.json').write_text('{"status":"RUNNING"}\n')
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0];lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))]
with tempfile.TemporaryDirectory(prefix='l103-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 checked=0
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/tgat-time-attention.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists(),(path,url);assert not dest.is_symlink();checked+=1
 for rel in ['labs/relkit/tgat_l103.py','labs/relkit/tgat_comparison_l103.py','labs/_verify_l103.py','labs/_fetch_l103.py','labs/solutions/'+S+'.ipynb','modal/l103_paper_repro.py']:
  assert (stage/rel).exists(),rel
 assert not (stage/'labs/sources/l103/original').exists()
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page()
   page.goto(f'http://127.0.0.1:{server.server_port}/index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]')
   assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(f'http://127.0.0.1:{server.server_port}/lessons/{S}.html');assert 'Baseline at t=4:' in page.locator('.tgat-readout').inner_text();browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
report={'status':'PASS','browser_widths':[1200,375],'widget_states':states,'independent_widget_arithmetic':'PASS','reset_and_keyboard':'PASS','print':'CHECKED','portable_figures':figures,'student_live_tasks':3,'copied_pages_local_links':checked,'deterministic_rebuild':'EXACT','manifest_navigation_over_http':'PASS','unlicensed_originals_excluded':'PASS','javascript_errors':errors,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
report['no_javascript_reading']='PASS at 1200 and 375 pixels'
(P/'_delivery_l103_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
