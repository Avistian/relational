"""Rendered lesson, standalone notebook and actual copied Pages verification."""
from _gallery_delivery import reveal_gallery_link
import ast,base64,hashlib,json,os,re,subprocess,sys,tempfile,functools,threading
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0139-healthcare-trial'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert all(not c.outputs for c in student.cells if c.cell_type=='code')
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');digest=hashlib.sha256(code.encode()).hexdigest()
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert digest==json.loads((P/'_execution_l139_results.json').read_text())['executed_code_sha256']
pinned=json.loads((P/'_notebook_pinned_l139_results.json').read_text());assert pinned['status']=='PASS' and pinned['code_sha256']==digest
fresh=json.loads((P/'_notebook_full_l139_results.json').read_text());assert fresh['status']=='PASS' and fresh['code_sha256']==digest
assert fresh['preparation']['source_sha256']==sha(P/'_full_l139.py')
assert 'from relkit' not in code
# The notebook must expose the exact executed full trainer and model primitives.
inline={n.name:n for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for source,names in [(P/'_full_l139.py',['full_run','materialize','sha']),(P/'relkit/rdl_l117.py',['Model','HeteroEncoder','HeteroTemporalEncoder','HeteroGraphSAGE','make_pkey_fkey_graph','get_node_train_table_input'])]:
 source_nodes={n.name:n for n in ast.parse(source.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 for name in names:assert ast.dump(inline[name],include_attributes=False)==ast.dump(source_nodes[name],include_attributes=False),name
for name,node in {n.name:n for n in ast.parse((P/'relkit/trial_l139.py').read_text()).body if isinstance(n,ast.FunctionDef)}.items():
 live=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name==name);assert ast.dump(node,include_attributes=False)==ast.dump(live,include_attributes=False)
images=re.findall('data:image/png;base64,([A-Za-z0-9+/=]+)','\n'.join(c.source for c in student.cells));assert len(images)==5
for data in images:assert base64.b64decode(data).startswith(b'\x89PNG')
paths=[R/'lessons'/f'{S}.html',R/'reference/healthcare-trial.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb'];before=[sha(p) for p in paths]
subprocess.run([sys.executable,str(P/'_build_l139.py')],check=True,capture_output=True);subprocess.run([sys.executable,str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True);assert before==[sha(p) for p in paths],'Builder drift'
figs=sorted((P/'figures/l139').glob('*'));before=[sha(p) for p in figs];subprocess.run([sys.executable,str(P/'_figures_l139.py')],check=True,capture_output=True);assert before==[sha(p) for p in figs]
exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True);html,_=exporter.from_notebook_node(solution);(P/'html'/f'{S}.html').write_text(html)
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size(dict(width=width,height=900));page.goto((R/'lessons'/f'{S}.html').as_uri())
  for name,values in [('target',[0,1,90,364,365,366]),('cutoff',[5,10,11,15,20,21,25])]:
   box=page.locator('#l139-'+name);control=box.locator('input')
   for value in values:
    control.fill(str(value));control.dispatch_event('input');text=box.locator('.audit-result').inner_text()
    if name=='target':assert ('Eligible: yes. Label: 1.' if 0<value<=365 else 'Eligible: no. Label: absent (not 0).')==text
    else:assert f"Query A: {'visible' if value<=10 else 'hidden'}. Query B: {'visible' if value<=20 else 'hidden'}."==text
    states+=1
   box.locator('button').click();assert control.input_value()==('365' if name=='target' else '15')
   control.focus();page.keyboard.press('ArrowRight');assert control.input_value()==('366' if name=='target' else '16');box.locator('button').click()
  assert page.locator('#warmup button').count()==0
  page.locator('figure img').evaluate_all('(xs)=>xs.forEach(x=>x.loading="eager")');page.wait_for_function('Array.from(document.querySelectorAll("figure img")).every(x=>x.complete && x.naturalWidth>0)');assert page.locator('figure img').evaluate_all('(xs)=>xs.length===4&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
  page.screenshot(path=f'/tmp/l139-top-{width}.png')
  for i in range(4):page.locator('figure').nth(i).screenshot(path=f'/tmp/l139-figure-{i}-{width}.png')
  page.locator('#l139-target').screenshot(path=f'/tmp/l139-window-{width}.png')
 page.emulate_media(media='print');page.screenshot(path='/tmp/l139-print.png',full_page=True);page.emulate_media(media='screen')
 page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('figure img[src^="data:image/png;base64,"]').count()==4;assert page.locator('img[src^="data:image/png;base64,"]').count()==5
 nojs=browser.new_context(java_script_enabled=False,viewport=dict(width=375,height=900));pg=nojs.new_page();pg.goto((R/'lessons'/f'{S}.html').as_uri());assert 'day365' in pg.locator('noscript').first.inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');nojs.close();browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0];lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))];checked=0
with tempfile.TemporaryDirectory(prefix='l139-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/healthcare-trial.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/trial_l139.py','labs/l139-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l139/queries.npz','labs/_full_l139.py','modal/l139_repro.py']:
  assert (stage/rel).exists(),rel
 manifest=json.loads((stage/'lessons/manifest.json').read_text());assert next(x for x in manifest['lessons'] if x['id']==139)['labPath']=='labs/'+S+'.ipynb'
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();base=f'http://127.0.0.1:{server.server_port}'
   page.goto(base+'/index.html');reveal_gallery_link(page, 'a[href="lessons/'+S+'.html"]');assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
   page.goto(base+'/notebooks.html');reveal_gallery_link(page, 'a[href="labs/html/'+S+'.html"]');browser.close()
 finally:server.shutdown();server.server_close();thread.join()

assert not errors,errors
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',print_nojs='PASS',portable_figures=4,portable_images=5,live_tasks=3,executed_code_hash=digest,deterministic_rebuild='EXACT',copied_pages_links=checked,javascript_errors=errors,screenshots='/tmp/l139-*.png',fresh_full_notebook='PASS',live_colab='NOT_CHECKED',deployment='NOT_CHECKED');(P/'_delivery_l139_results.json').write_text(json.dumps(r,indent=2));print(r)
