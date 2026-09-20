"""Browser interactions, portable notebooks, deterministic builds and copied Pages staging."""
import functools,hashlib,json,os,re,subprocess,tempfile,threading,math
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0100-heterogeneous-gnn-checkpoint'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
errors=[]
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
 page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri())
  widget=page.locator('#batch-weighting');select=widget.locator('select');out=widget.locator('output')
  assert 'Naive batch mean: 0.625' in out.inner_text()
  for size in range(1,6):
   select.select_option(str(size));losses=[.2,.8,.5,1.1,.4];groups=[losses[i:i+size] for i in range(0,5,size)]
   naive=sum(sum(g)/len(g) for g in groups)/len(groups)
   assert f'Naive batch mean: {naive:.3f}' in out.inner_text() and 'Weighted mean: 0.600' in out.inner_text()
  widget.locator('button').click();assert select.input_value()=='3'
  select.focus();page.keyboard.press('ArrowDown');assert select.input_value()=='4';widget.locator('button').click()
  assert page.locator('#l100-warmup').inner_text().strip()
  pred=page.locator('#l100-predict');assert pred.locator('button').last.is_disabled()
  pred.locator('button').nth(1).click();pred.locator('button').last.click()
  teach=page.locator('#l100-teachback');teach.locator('textarea').fill('The architecture comparison changes attention, residuals and normalization. A paired uniform-attention intervention narrows the mechanism claim, but capacity and optimization still change.');teach.locator('button').first.click()
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Page overflow'
  for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
  page.evaluate('window.scrollTo(0,0)');page.screenshot(path=f'/tmp/l100-top-{width}.png');widget.screenshot(path=f'/tmp/l100-widget-{width}.png');page.locator('figure').first.screenshot(path=f'/tmp/l100-architecture-{width}.png')
 page.emulate_media(media='print');assert page.locator('h1').is_visible();page.emulate_media(media='screen')
 for name in ['architecture','identity','weighting']:
  page.goto((P/f'figures/l100/{name}.svg').as_uri())
  assert page.locator('svg').evaluate("s=>{let v=s.viewBox.baseVal;return Array.from(s.querySelectorAll('text')).every(t=>{let b=t.getBBox();return b.x>=0&&b.y>=0&&b.x+b.width<=v.width&&b.y+b.height<=v.height})}"),name+' SVG label exceeds viewBox'
 page.goto((P/'html'/f'{S}.html').as_uri())
 for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
 assert page.locator('img').count()>=1
 browser.close()
student=nbformat.read(P/f'{S}.ipynb',as_version=4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
assert sum('raise NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==3
assert not any('raise NotImplementedError' in c.source for c in solution.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert 'data:image/png;base64,' in json.dumps(solution) and 'attachment:' not in json.dumps(solution)
paths=[R/'lessons'/f'{S}.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb',P/'html'/f'{S}.html']
def hashes():return [hashlib.sha256(x.read_bytes()).hexdigest() for x in paths]
before=hashes();subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l100.py')],check=True,capture_output=True);assert hashes()==before,'Rebuild changed artifacts'
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.ids=set()
 def handle_starttag(self,tag,attrs):
  self.links.extend(v for k,v in attrs if k in ('href','src'))
  self.ids.update(v for k,v in attrs if k=='id')
(P/'_delivery_l100_results.json').write_text('{"status":"RUNNING"}\n')
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0];lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))]
with tempfile.TemporaryDirectory(prefix='l100-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 checked=0
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/heterogeneous-checkpoint.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists(),(path,url);assert not dest.is_symlink();checked+=1
   if part.fragment and dest.suffix=='.html':
    other=Links();other.feed(dest.read_text());assert any(unquote(i)==unquote(part.fragment) for i in other.ids),(dest,part.fragment)
 for relative in ['labs/relkit/checkpoint_l100.py','labs/_run_l100.py','labs/l100-reproduction.md','labs/_experiment_l100_results.json','labs/_paper_l100_aifb_results.json','labs/solutions/'+S+'.ipynb','labs/_experiment_l100_results-checkpoints/hgt-0.003-0.pt']:
  assert (stage/relative).exists(),relative
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)))
 thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
   page.goto(f'http://127.0.0.1:{server.server_port}/index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]')
   assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(f'http://127.0.0.1:{server.server_port}/lessons/{S}.html');assert 'Weighted mean: 0.600' in page.locator('#batch-weighting output').inner_text()
   browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
report={'status':'PASS','browser_widths':[1200,375],'all_five_batch_sizes_reset_keyboard':'PASS','print':'CHECKED','portable_figures':4,'solution_code_cells':sum(c.cell_type=='code' for c in solution.cells),'student_live_tasks':3,'copied_pages_local_links':checked,'deterministic_rebuild':'EXACT','manifest_navigation_over_http':'PASS','javascript_errors':errors,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/'_delivery_l100_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
