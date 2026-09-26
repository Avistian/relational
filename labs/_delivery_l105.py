"""Verify the actual lesson/notebook, interactive arithmetic, and copied Pages site."""
import functools,hashlib,json,os,re,subprocess,sys,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0105-continuous-time'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((P/'_analysis_l105_results.json').read_text())
for name,h in report['code_sha256'].items():assert sha(P/name)==h,name
for name,h in report['artifact_sha256'].items():assert sha(P/'evidence/l105'/name)==h,name
student=nbformat.read(P/f'{S}.ipynb',as_version=4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
assert sum('raise NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==3
assert all(not c.outputs for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
source='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert 'from relkit' not in source and 'RUN_COMPLETE' not in source
assert hashlib.sha256(source.encode()).hexdigest()==json.loads((P/'_execution_l105_results.json').read_text())['executed_code_sha256']
# Every canonical code chunk is visible unchanged in solution, not a hidden import.
canonical=(P/'relkit/stream_l105.py').read_text()
for chunk in re.split(r'^# %% ',canonical,flags=re.M)[1:]:assert chunk.split('\n',1)[1].strip() in source
# Rebuild without rewriting executed output; static outputs must be byte-identical.
paths=[R/'lessons'/f'{S}.html',R/'reference/event-stream-snapshots.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb']
before=[sha(p) for p in paths]
subprocess.run([sys.executable,str(P/'_build_l105.py')],check=True,capture_output=True)
assert before==[sha(p) for p in paths],'nondeterministic lesson/notebook rebuild'
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
 page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('requestfailed',lambda r:errors.append(r.url))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri())
  agg=page.locator('#l105-aggregation');select=agg.locator('select')
  for window,edge_count in [('5',5),('10',5),('20',4)]:
   select.select_option(window);cards=agg.locator('.stream-cards strong');assert cards.all_text_contents()==['6',str(edge_count),'6'];states+=1
  agg.locator('button').click();assert select.input_value()=='10'
  clock=page.locator('#l105-release');slider=clock.locator('input[type=range]');late=clock.locator('input[type=checkbox]')
  times=[1,3,3,8,12,19]
  for delayed in [False,True]:
   late.set_checked(delayed);late.dispatch_event('input')
   for q in range(23):
    slider.fill(str(q));slider.dispatch_event('input')
    eligible=sum(t<q and (15 if delayed and t==8 else t)<=q for t in times)
    first_release=15 if delayed else 10;released=(4 if q>=first_release else 0)+(2 if q>=20 else 0)
    assert clock.locator('.stream-cards strong').all_text_contents()==list(map(str,[eligible,released,eligible-released]))
    nonpast=sum(t>=q and t//10==q//10 for t in times)
    assert f'expose {nonpast} nonpast' in clock.locator('.stream-live').inner_text();states+=1
  clock.locator('button').click();assert slider.input_value()=='6' and not late.is_checked()
  slider.focus();page.keyboard.press('ArrowRight');assert slider.input_value()=='7';clock.locator('button').click()
  pred=page.locator('#l105-predict');assert pred.locator('button').last.is_disabled();pred.locator('button').nth(1).click();pred.locator('button').last.click();assert 'identical endpoint counts' in pred.inner_text()
  teach=page.locator('#l105-teachback');teach.locator('textarea').fill('I retain event time and availability for immediate prediction. Count-weighted windows preserve pair totals, but discard within-window order. At query 10 the window [0,10) is legal only when all records have arrived. A complete current window at noon can contain future events. The audit measures representation counts, not predictive performance.');teach.locator('button').first.click()
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'page overflow'
  for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
  page.screenshot(path=f'/tmp/l105-full-{width}.png',full_page=True)
  agg.screenshot(path=f'/tmp/l105-aggregation-{width}.png');clock.screenshot(path=f'/tmp/l105-release-{width}.png')
  page.locator('figure').nth(1).screenshot(path=f'/tmp/l105-order-{width}.png')
 page.emulate_media(media='print');assert page.locator('details p').first.evaluate('(e)=>e.checkVisibility()');assert not page.locator('#l105-aggregation').is_visible()
 page.screenshot(path='/tmp/l105-print.png',full_page=True);page.emulate_media(media='screen')
 for name in ['aggregation','order','release','results']:
  page.goto((P/f'figures/l105/{name}.svg').as_uri())
  assert page.locator('svg').evaluate("s=>{const v=s.viewBox.baseVal;return Array.from(s.querySelectorAll('text')).every(t=>{const b=t.getBoundingClientRect(),r=s.getBoundingClientRect();return b.x>=r.x-1&&b.y>=r.y-1&&b.right<=r.right+1&&b.bottom<=r.bottom+1})}"),name+' label clipping'
 page.goto((P/'html'/f'{S}.html').as_uri());figures=page.locator('img[src^="data:image/png;base64,"]').count();assert figures==4
 page.set_viewport_size({'width':950,'height':900});page.locator('img[src^="data:image/png;base64,"]').nth(2).screenshot(path='/tmp/l105-notebook-release.png')
 nojs=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});plain=nojs.new_page();plain.goto((R/'lessons'/f'{S}.html').as_uri())
 assert 'Without opening L104' in plain.locator('article').inner_text() and plain.locator('figure').count()==4
 assert plain.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
 plain.emulate_media(media='print');assert plain.locator('details p').first.evaluate('(e)=>e.checkVisibility()')
 nojs.close();browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
(P/'_delivery_l105_results.json').write_text('{"status":"RUNNING"}\n')
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))]
with tempfile.TemporaryDirectory(prefix='l105-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 checked=0
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/event-stream-snapshots.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/stream_l105.py','labs/_run_l105.py','labs/l105-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l105/events.npz','labs/sources/l105/tgn_preprocess_data.py.txt']:
  assert (stage/rel).exists(),rel
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
   base=f'http://127.0.0.1:{server.server_port}'
   page.goto(base+'/index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]');assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(base+'/notebooks.html');page.wait_for_selector('a[href="labs/html/'+S+'.html"]')
   page.goto(base+f'/lessons/{S}.html');assert 'repeated records collapse' in page.locator('#l105-aggregation .stream-live').inner_text();browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
visual_path=P/'_visual_l105_results.json'
visual='PENDING'
if visual_path.exists():
 v=json.loads(visual_path.read_text())
 for name,h in v['reviewed_sha256'].items():assert sha(R/name)==h, 'Visual review stale: '+name
 visual=v['status']
out={'status':'PASS','browser_widths':[1200,375],'widget_states':states,'reset_and_keyboard':'PASS','print':'PASS','no_js':'PASS','portable_figures':figures,'student_live_tasks':3,'canonical_inline_source':'EXACT','executed_code_hash':'MATCH','artifact_and_source_hashes':'MATCH','copied_pages_links':checked,'deterministic_rebuild':'EXACT','manifest_navigation_over_http':'PASS','javascript_errors':errors,'screenshots':'/tmp/l105-*.png','visual_review':visual,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/'_delivery_l105_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
