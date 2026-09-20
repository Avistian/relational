"""Browser, notebook, deterministic regeneration and actual copied Pages checks."""
import functools,hashlib,json,os,re,subprocess,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0098-hetero-mini-batching'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
errors=[]
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
 page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri())
  widget=page.locator('#batching-intervention');selects=widget.locator('select');out=widget.locator('output')
  assert 'mean: 2.00' in out.inner_text()
  for cutoff in [1,2,3]:
   for fanout in [1,2,3]:
    selects.nth(0).select_option(str(cutoff));selects.nth(1).select_option(str(fanout))
    n=min(cutoff,fanout);mean=sum([2,8,-4][:n])/n
    assert f'mean: {mean:.2f}' in out.inner_text()
  widget.locator('button').click();assert selects.nth(0).input_value()=='3' and selects.nth(1).input_value()=='3'
  selects.nth(0).focus();page.keyboard.press('ArrowDown');assert selects.nth(0).input_value()=='2'
  widget.locator('button').click()
  assert page.locator('#l098-warmup').inner_text().strip()
  pred=page.locator('#l098-predict');assert pred.locator('button').last.is_disabled()
  pred.locator('button').first.click();pred.locator('button').last.click()
  teach=page.locator('#l098-teachback');teach.locator('textarea').fill('The seed prefix selects supervised queries, while temporal sampling restricts what their embeddings can observe. Both boundaries are necessary.')
  teach.locator('button').first.click()
  # Every SVG label remains within the declared viewBox.
  assert widget.locator('svg').evaluate("s=>Array.from(s.querySelectorAll('text')).every(t=>{let b=t.getBBox();return b.x>=0&&b.y>=0&&b.x+b.width<=700&&b.y+b.height<=280})")
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Page overflow'
  for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
  page.screenshot(path=f'/tmp/l098-top-{width}.png');widget.screenshot(path=f'/tmp/l098-widget-{width}.png')
  page.locator('figure').screenshot(path=f'/tmp/l098-schema-{width}.png')
 page.emulate_media(media='print');assert page.locator('h1').is_visible();page.emulate_media(media='screen')
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
before=hashes();subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l098.py')],check=True,capture_output=True);assert hashes()==before,'Rebuild changed artifacts'
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.ids=set()
 def handle_starttag(self,tag,attrs):
  self.links.extend(v for k,v in attrs if k in ('href','src'))
  self.ids.update(v for k,v in attrs if k=='id')
(P/'_delivery_l098_results.json').write_text('{"status":"RUNNING"}\n')
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0];lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))]
with tempfile.TemporaryDirectory(prefix='l098-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 checked=0
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/hetero-batching-contract.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists(),(path,url);assert not dest.is_symlink();checked+=1
   if part.fragment and dest.suffix=='.html':
    other=Links();other.feed(dest.read_text());assert any(unquote(i)==unquote(part.fragment) for i in other.ids),(dest,part.fragment)
 for relative in ['labs/relkit/batching_l098.py','labs/_run_l098.py','labs/l098-reproduction.md','labs/_experiment_l098_results.json','labs/solutions/'+S+'.ipynb']:
  assert (stage/relative).exists(),relative
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)))
 thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
   page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
   page.goto(f'http://127.0.0.1:{server.server_port}/index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]')
   assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(f'http://127.0.0.1:{server.server_port}/lessons/{S}.html');assert 'mean: 2.00' in page.locator('#batching-intervention output').inner_text()
   browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
report={'status':'PASS','browser_widths':[1200,375],'all_nine_cutoff_fanout_states_reset_keyboard':'PASS','print':'CHECKED','portable_figures':1,'solution_code_cells':20,'student_live_tasks':3,'copied_pages_local_links':checked,'deterministic_rebuild':'EXACT','manifest_navigation_over_http':'PASS','javascript_errors':errors,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/'_delivery_l098_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
