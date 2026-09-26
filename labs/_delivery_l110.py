"""Real browser, notebook and copied deployment-tree checks; separate from scientific evidence."""
import argparse,ast,functools,hashlib,json,os,re,subprocess,tempfile,threading,sys
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');args=p.parse_args()
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0110-temporal-gnn-checkpoint'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
student=nbformat.read(P/f'{S}.ipynb',as_version=4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert not any(c.outputs for c in student.cells if c.cell_type=='code')
source='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert sha(P/'relkit/checkpoint_l110.py')==json.loads((P/'_sources_l110.json').read_text())['implementation_sha256']
assert hashlib.sha256(source.encode()).hexdigest()==json.loads((P/'_execution_l110_results.json').read_text())['executed_code_sha256']
assert 'from relkit' not in source
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
def definitions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
canonical=definitions((P/'relkit/checkpoint_l110.py').read_text());sol=definitions(source);stu=definitions('\n\n'.join(c.source for c in student.cells if c.cell_type=='code'))
for name,node in canonical.items():
 assert sol[name]==node,name
 if name not in ['restore_checkpoint','strict_batches','legal_history']:assert stu[name]==node,name
paths=[R/'lessons'/f'{S}.html',R/'reference/temporal-gnn-checkpoint.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb']
before=[sha(x) for x in paths];subprocess.run([sys.executable,str(P/'_build_l110.py')],check=True,capture_output=True);assert before==[sha(x) for x in paths],'Rebuild drift'
figures=sorted((P/'figures/l110').glob('*'));before=[sha(x) for x in figures];subprocess.run([sys.executable,str(P/'_figures_l110.py')],check=True,capture_output=True);assert before==[sha(x) for x in figures],'Figure drift'
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('requestfailed',lambda r:errors.append(r.url))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri());widget=page.locator('#l110-batches');slider=widget.locator('input[type=range]');toggle=widget.locator('input[type=checkbox]')
  assert '1 tied boundaries' in widget.inner_text()
  for size in range(1,7):
   for clean in [False,True]:
    slider.fill(str(size));slider.dispatch_event('input');toggle.set_checked(clean)
    ts=[1,2,2,2,3,4];groups=[];i=0
    while i<len(ts):
     j=min(i+size,len(ts))
     if clean:
      while j<len(ts) and ts[j]==ts[j-1]:j+=1
     groups.append(ts[i:j]);i=j
    expected=sum(a[-1]==b[0] for a,b in zip(groups,groups[1:]));assert f'{expected} tied boundaries' in widget.locator('[data-verdict]').inner_text();states+=1
  widget.locator('button').click();assert slider.input_value()=='2' and not toggle.is_checked();slider.focus();page.keyboard.press('ArrowRight');assert slider.input_value()=='3';widget.locator('button').click();widget.screenshot(path=f'/tmp/l110-widget-{width}.png')
  pred=page.locator('#l110-predict');assert pred.locator('button').last.is_disabled();pred.locator('button').first.click();pred.locator('button').last.click();assert 'Consistent history' in pred.inner_text()
  teach=page.locator('#l110-teachback');assert teach.locator('button').first.is_disabled();teach.locator('textarea').fill('The sampler controls adjacency but memory is a separate information path. Queued messages must come from strictly earlier timestamp groups. Restore weights, memory, clocks and pending messages from one selected epoch before each test branch. Event timestamps do not prove actual availability.');teach.locator('button').first.click();assert 'queued messages' in teach.inner_text()
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),'Page overflow';assert page.locator('figure').count()==4;page.screenshot(path=f'/tmp/l110-page-{width}.png',full_page=True)
 for name in ['architecture','checkpoint','ties','results']:
  page.goto((P/f'figures/l110/{name}.svg').as_uri());assert page.locator('svg').evaluate("s=>{const r=s.getBoundingClientRect();return Array.from(s.querySelectorAll('text')).every(t=>{const b=t.getBoundingClientRect();return b.x>=r.x-1&&b.y>=r.y-1&&b.right<=r.right+1&&b.bottom<=r.bottom+1})}"),name+' labels outside canvas'
 page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('img[src^="data:image/png;base64,"]').count()==4;page.set_viewport_size({'width':950,'height':900})
 for i in range(4):page.locator('img[src^="data:image/png;base64,"]').nth(i).screenshot(path=f'/tmp/l110-notebook-{i}.png')
 nojs=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});plain=nojs.new_page();plain.goto((R/'lessons'/f'{S}.html').as_uri());assert plain.locator('figure').count()==4 and 'Worked answer, including a no-JavaScript fallback' in plain.locator('article').inner_text();assert plain.evaluate('document.documentElement.scrollWidth<=innerWidth+1');plain.emulate_media(media='print');assert plain.locator('details p').first.evaluate('(e)=>e.checkVisibility()');plain.screenshot(path='/tmp/l110-print.png',full_page=True);nojs.close();browser.close()
checked=0
if not args.preview:
 assert json.loads((P/'evidence/l110/summary.json').read_text())['status']=='COMPLETE'
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.links=[]
  def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
 (P/'_delivery_l110_results.json').write_text('{"status":"RUNNING"}\n')
 workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0];lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))]
 with tempfile.TemporaryDirectory(prefix='l110-pages-') as tmp:
  stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines);subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
  for path in [stage/'lessons'/f'{S}.html',stage/'reference/temporal-gnn-checkpoint.html']:
   parser=Links();parser.feed(path.read_text())
   for url in parser.links:
    part=urlsplit(url)
    if part.scheme:continue
    dest=(path.parent/unquote(part.path)).resolve() if part.path else path;assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
  for rel in ['labs/relkit/checkpoint_l110.py','labs/_run_l110.py','labs/l110-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l110/summary.json','labs/sources/l102/model/tgn.py','modal/l110_repro.py']:assert (stage/rel).exists(),rel
  class Quiet(SimpleHTTPRequestHandler):
   def log_message(self,*args):pass
  server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));base=f'http://127.0.0.1:{server.server_port}'
    page.goto(base+'/index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]');assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
    page.goto(base+'/notebooks.html');page.wait_for_selector('a[href="labs/html/'+S+'.html"]');page.goto(base+'/lessons/'+S+'.html');assert '1 tied boundaries' in page.locator('#l110-batches').inner_text();browser.close()
  finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
out={'status':'PREVIEW_PASS' if args.preview else 'PASS','browser_widths':[1200,375],'widget_states':states,'reset_keyboard':'PASS','prediction_and_teachback':'PASS','print_nojs':'PASS','portable_figures':4,'student_live_tasks':3,'canonical_definitions':len(canonical),'executed_code_hash':'MATCH','deterministic_rebuild':'EXACT','copied_pages_links':checked,'javascript_errors':errors,'screenshots':'/tmp/l110-*.png','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/('_preview_l110_results.json' if args.preview else '_delivery_l110_results.json')).write_text(json.dumps(out,indent=2));print(out)
