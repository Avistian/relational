"""Real browser interaction, portable notebooks, deterministic builds and copied Pages."""
import functools,hashlib,json,os,re,subprocess,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0101-static-vs-temporal'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
errors=[]
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
 page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri())
  widget=page.locator('#temporal-visibility');slider=widget.locator('input');select=widget.locator('select');out=widget.locator('.temporal-readout')
  assert 'Legal mean: 3.000' in out.inner_text() and 'Compared mean: 4.667' in out.inner_text()
  rows=[(3,3,2),(4,7,8),(5,5,4),(8,8,10)]
  for mode in ['event','static']:
   select.select_option(mode)
   for day in range(11):
    slider.fill(str(day));slider.dispatch_event('input')
    legal=[v for e,a,v in rows if max(e,a)<=day];other=[v for e,a,v in rows if mode=='static' or e<=day]
    mean=lambda a:sum(a)/len(a) if a else 0
    assert f'Legal mean: {mean(legal):.3f}' in out.inner_text()
    assert f'Compared mean: {mean(other):.3f}' in out.inner_text()
  widget.locator('button').click();slider.focus();page.keyboard.press('ArrowRight');assert slider.input_value()=='6';widget.locator('button').click()
  assert page.locator('#l101-warmup').inner_text().strip()
  pred=page.locator('#l101-predict');assert pred.locator('button').last.is_disabled();pred.locator('button').nth(1).click();pred.locator('button').last.click()
  teach=page.locator('#l101-teachback');teach.locator('textarea').fill('Every sampled dependency must have existed at the query cutoff, including feature versions. Every training label must mature by the fit cutoff. Chronological targets alone do not enforce either graph availability or versioned attributes.');teach.locator('button').first.click()
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'page overflow'
  for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
  page.evaluate('window.scrollTo(0,0)');page.screenshot(path=f'/tmp/l101-top-{width}.png');widget.screenshot(path=f'/tmp/l101-widget-{width}.png');page.locator('figure').first.screenshot(path=f'/tmp/l101-path-{width}.png')
 page.emulate_media(media='print');assert page.locator('h1').is_visible();page.emulate_media(media='screen')
 for name in ['query-path','label-window']:
  page.goto((P/f'figures/l101/{name}.svg').as_uri())
  assert page.locator('svg').evaluate("s=>{let v=s.viewBox.baseVal;return Array.from(s.querySelectorAll('text')).every(t=>{let b=t.getBBox();return b.x>=0&&b.y>=0&&b.x+b.width<=v.width&&b.y+b.height<=v.height})}"),name+' label outside figure'
 page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('img[src^="data:image/png;base64,"]').count()==2
 browser.close()
student=nbformat.read(P/f'{S}.ipynb',as_version=4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
assert sum('raise NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==3
assert all(not c.outputs for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert 'from relkit' not in '\n'.join(c.source for c in solution.cells)
paths=[R/'lessons'/f'{S}.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb']
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l101.py')],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'nondeterministic build'
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
(P/'_delivery_l101_results.json').write_text('{"status":"RUNNING"}\n')
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0];lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))]
with tempfile.TemporaryDirectory(prefix='l101-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 checked=0
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/temporal-visibility.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists(),(path,url);assert not dest.is_symlink();checked+=1
 for rel in ['labs/relkit/temporal_l101.py','labs/_verify_l101.py','labs/_predictions_l101.npz','labs/sources/l101/baseline_node.py','labs/solutions/'+S+'.ipynb']:
  assert (stage/rel).exists(),rel
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page()
   page.goto(f'http://127.0.0.1:{server.server_port}/index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]')
   assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(f'http://127.0.0.1:{server.server_port}/lessons/{S}.html');assert 'Legal mean: 3.000' in page.locator('.temporal-readout').inner_text();browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
report={'status':'PASS','browser_widths':[1200,375],'widget_states':44,'reset_and_keyboard':'PASS','print':'CHECKED','portable_figures':2,'student_live_tasks':3,'copied_pages_local_links':checked,'deterministic_rebuild':'EXACT','manifest_navigation_over_http':'PASS','javascript_errors':errors,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/'_delivery_l101_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
