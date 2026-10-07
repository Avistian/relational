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
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0138-ecommerce-amazon'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert all(not c.outputs for c in student.cells if c.cell_type=='code')
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');digest=hashlib.sha256(code.encode()).hexdigest()
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert digest==json.loads((P/'_execution_l138_results.json').read_text())['executed_code_sha256']
pinned=json.loads((P/'_notebook_pinned_l138_results.json').read_text());assert pinned['status']=='PASS' and pinned['code_sha256']==digest
assert 'from relkit' not in code
# The notebook must expose the exact executed full trainer and model primitives.
inline={n.name:n for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for source,names in [(P/'_full_l138.py',['full_run']),(P/'relkit/rdl_l117.py',['Model','HeteroEncoder','HeteroTemporalEncoder','HeteroGraphSAGE','make_pkey_fkey_graph','get_node_train_table_input'])]:
 source_nodes={n.name:n for n in ast.parse(source.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 for name in names:assert ast.dump(inline[name],include_attributes=False)==ast.dump(source_nodes[name],include_attributes=False),name
for name,node in {n.name:n for n in ast.parse((P/'relkit/amazon_l138.py').read_text()).body if isinstance(n,ast.FunctionDef)}.items():
 live=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name==name);assert ast.dump(node,include_attributes=False)==ast.dump(live,include_attributes=False)
images=re.findall('data:image/png;base64,([A-Za-z0-9+/=]+)','\n'.join(c.source for c in student.cells));assert len(images)==4
for data in images:assert base64.b64decode(data).startswith(b'\x89PNG')
paths=[R/'lessons'/f'{S}.html',R/'reference/ecommerce-amazon.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb'];before=[sha(p) for p in paths]
subprocess.run([sys.executable,str(P/'_build_l138.py')],check=True,capture_output=True);subprocess.run([sys.executable,str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True);assert before==[sha(p) for p in paths],'Builder drift'
figs=sorted((P/'figures/l138').glob('*'));before=[sha(p) for p in figs];subprocess.run([sys.executable,str(P/'_figures_l138.py')],check=True,capture_output=True);assert before==[sha(p) for p in figs]
exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True);html,_=exporter.from_notebook_node(solution);(P/'html'/f'{S}.html').write_text(html)
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size(dict(width=width,height=900));page.goto((R/'lessons'/f'{S}.html').as_uri())
  for name,values in [('window',[0,1,90,91,92,100]),('path',[0,1,2])]:
   box=page.locator('#l138-'+name);control=box.locator('input')
   for value in values:
    control.fill(str(value));control.dispatch_event('input');text=box.locator('.audit-result').inner_text()
    if name=='window':assert f'Churn label: {int(not(0<value<=91))}' in text
    else:assert ('Hop 1: r0.' if value==0 else 'Hop 1: r0, r1.') in text
    states+=1
   box.locator('button').click();assert control.input_value()==('91' if name=='window' else '0')
   control.focus();page.keyboard.press('ArrowRight');assert control.input_value()==('92' if name=='window' else '1');box.locator('button').click()
  assert page.locator('#warmup button').count()==0
  page.locator('figure img').evaluate_all('(xs)=>xs.forEach(x=>x.loading="eager")');page.wait_for_function('Array.from(document.querySelectorAll("figure img")).every(x=>x.complete && x.naturalWidth>0)');assert page.locator('figure img').evaluate_all('(xs)=>xs.length===3&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
  page.screenshot(path=f'/tmp/l138-top-{width}.png')
  for i in range(3):page.locator('figure').nth(i).screenshot(path=f'/tmp/l138-figure-{i}-{width}.png')
  page.locator('#l138-window').screenshot(path=f'/tmp/l138-window-{width}.png')
 page.emulate_media(media='print');page.screenshot(path='/tmp/l138-print.png',full_page=True);page.emulate_media(media='screen')
 page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('figure img[src^="data:image/png;base64,"]').count()==3;assert page.locator('img[src^="data:image/png;base64,"]').count()==4
 nojs=browser.new_context(java_script_enabled=False,viewport=dict(width=375,height=900));pg=nojs.new_page();pg.goto((R/'lessons'/f'{S}.html').as_uri());assert 'day+91' in pg.locator('noscript').first.inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');nojs.close();browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0];lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))];checked=0
with tempfile.TemporaryDirectory(prefix='l138-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/ecommerce-amazon.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/amazon_l138.py','labs/l138-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l138/recency_predictions.npz','labs/_full_l138.py','modal/l138_full.py']:
  assert (stage/rel).exists(),rel
 manifest=json.loads((stage/'lessons/manifest.json').read_text());assert next(x for x in manifest['lessons'] if x['id']==138)['labPath']=='labs/'+S+'.ipynb'
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
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',print_nojs='PASS',portable_figures=3,portable_images=4,live_tasks=3,executed_code_hash=digest,deterministic_rebuild='EXACT',copied_pages_links=checked,javascript_errors=errors,screenshots='/tmp/l138-*.png',live_colab='NOT_CHECKED',deployment='NOT_CHECKED');(P/'_delivery_l138_results.json').write_text(json.dumps(r,indent=2));print(r)
