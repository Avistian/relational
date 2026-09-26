"""Validate visible code, executed notebook, rendering and copied Pages deployment shape."""
import ast,functools,hashlib,json,os,re,subprocess,tempfile,threading,sys
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0109-database-timestamp-contracts'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
student=nbformat.read(P/f'{S}.ipynb',as_version=4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert not any(c.outputs for c in student.cells if c.cell_type=='code')
source='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert hashlib.sha256(source.encode()).hexdigest()==json.loads((P/'_execution_l109_results.json').read_text())['executed_code_sha256']
assert 'from relkit' not in source
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
definitions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};count=0
for name in ['database_l109.py','f1_l109.py']:
 for node in ast.parse((P/'relkit'/name).read_text()).body:
  if isinstance(node,(ast.FunctionDef,ast.ClassDef)):assert definitions[node.name]==ast.dump(node,include_attributes=False),node.name;count+=1
# The only function-body differences in the student notebook are the three live blanks.
student_defs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse('\n\n'.join(c.source for c in student.cells if c.cell_type=='code')).body if isinstance(n,ast.FunctionDef)}
for name,value in definitions.items():
 if name not in ['asof_versions','legal_history','label_ready']:assert student_defs[name]==value,name
paths=[R/'lessons'/f'{S}.html',R/'reference/database-timestamp-contracts.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb']
before=[sha(x) for x in paths];subprocess.run([sys.executable,str(P/'_build_l109.py')],check=True,capture_output=True);assert before==[sha(x) for x in paths],'Rebuild drift'
figures=sorted((P/'figures/l109').glob('*'));before=[sha(x) for x in figures];subprocess.run([sys.executable,str(P/'_figures_l109.py')],check=True,capture_output=True);assert before==[sha(x) for x in figures],'Figure drift'
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
 page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('requestfailed',lambda r:errors.append(r.url))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri());widget=page.locator('#l109-history');cut=widget.locator('input')
  assert widget.locator('[data-edge]').inner_text()=='r → A'
  for t in range(13):
   cut.fill(str(t));cut.dispatch_event('input');expected='No fact known' if t<3 else 'r → A' if t<8 else 'r → B' if t<10 else 'Fact deleted; no edge'
   assert widget.locator('[data-edge]').inner_text()==expected
   assert widget.locator('[data-value]').inner_text()==('—' if t<3 or t>=10 else '1' if t<8 else '2')
   assert widget.locator('[data-ready]').inner_text()==('—' if t<3 or t>=10 else '3' if t<8 else '8');states+=1
   assert 'day-6 baseline' in widget.locator('[data-baseline]').inner_text()
  widget.locator('button').click();assert cut.input_value()=='6';cut.focus();page.keyboard.press('ArrowRight');assert cut.input_value()=='7';widget.locator('button').click()
  widget.screenshot(path=f'/tmp/l109-widget-{width}.png')
  pred=page.locator('#l109-predict');assert pred.locator('button').last.is_disabled();pred.locator('button').first.click();pred.locator('button').last.click();assert 'ingestion' in pred.inner_text()
  teach=page.locator('#l109-teachback');assert teach.locator('button').first.is_disabled();teach.locator('textarea').fill('At cutoff six choose the effective and observed row versions, then join the selected foreign key to a historical parent. The deletion is selected before removing the row. Labels require a closed window and certified availability. The released F1 data cannot prove its unrecorded ingestion history.');teach.locator('button').first.click();assert 'tombstone' in teach.inner_text()
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),'Page overflow'
  assert page.locator('figure').count()==4
  page.screenshot(path=f'/tmp/l109-page-{width}.png',full_page=True)
 for name in ['clocks','versions','maturity','results']:
  page.goto((P/f'figures/l109/{name}.svg').as_uri());assert page.locator('svg').evaluate("s=>{const r=s.getBoundingClientRect();return Array.from(s.querySelectorAll('text')).every(t=>{const b=t.getBoundingClientRect();return b.x>=r.x-1&&b.y>=r.y-1&&b.right<=r.right+1&&b.bottom<=r.bottom+1})}"),name+' labels outside canvas'
 page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('img[src^="data:image/png;base64,"]').count()==4
 page.set_viewport_size({'width':950,'height':900})
 for i in range(4):page.locator('img[src^="data:image/png;base64,"]').nth(i).screenshot(path=f'/tmp/l109-notebook-{i}.png')
 nojs=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});plain=nojs.new_page();plain.goto((R/'lessons'/f'{S}.html').as_uri());assert plain.locator('figure').count()==4 and 'Worked answer, including a no-JavaScript fallback' in plain.locator('article').inner_text();assert plain.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
 plain.emulate_media(media='print');assert plain.locator('details p').first.evaluate('(e)=>e.checkVisibility()');plain.screenshot(path='/tmp/l109-print.png',full_page=True);nojs.close();browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
(P/'_delivery_l109_results.json').write_text('{"status":"RUNNING"}\n')
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))]
with tempfile.TemporaryDirectory(prefix='l109-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True);checked=0
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/database-timestamp-contracts.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/database_l109.py','labs/relkit/f1_l109.py','labs/_verify_l109.py','labs/l109-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l109/predictions.npz','labs/sources/l109/task_f1.py']:
  assert (stage/rel).exists(),rel
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));base=f'http://127.0.0.1:{server.server_port}'
   page.goto(base+'/index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]');assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(base+'/notebooks.html');page.wait_for_selector('a[href="labs/html/'+S+'.html"]')
   page.goto(base+'/lessons/'+S+'.html');assert page.locator('#l109-history [data-edge]').inner_text()=='r → A';browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
out={'status':'PASS','browser_widths':[1200,375],'widget_states':states,'reset_keyboard':'PASS','prediction_and_teachback':'PASS','print_nojs':'PASS','portable_figures':4,'student_live_tasks':3,'canonical_definitions':count,'executed_code_hash':'MATCH','deterministic_rebuild':'EXACT','copied_pages_links':checked,'manifest_navigation':'PASS','javascript_errors':errors,'screenshots':'/tmp/l109-*.png','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/'_delivery_l109_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
