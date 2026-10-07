"""Standalone execution, browser states and copied Pages checks for L140."""
import ast,base64,hashlib,json,os,re,subprocess,sys,tempfile,functools,threading
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0140-rdl-reproduction-checkpoint'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert all(not c.outputs for c in student.cells if c.cell_type=='code')
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');digest=hashlib.sha256(code.encode()).hexdigest()
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert digest==json.loads((P/'_execution_l140_results.json').read_text())['executed_code_sha256']
pinned=json.loads((P/'_notebook_trial_l140_results.json').read_text());assert pinned['status']=='PASS' and pinned['code_sha256']==digest
assert 'from relkit' not in code
inline={n.name:n for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for source,names in [(P/'_full_l140.py',['full_run','materialize','sha']),(P/'relkit/rdl_l117.py',['Model','HeteroEncoder','HeteroTemporalEncoder','HeteroGraphSAGE','make_pkey_fkey_graph','get_node_train_table_input']),(P/'relkit/checkpoint_l140.py',['select_checkpoint','align_predictions','reproduction_verdict'])]:
 source_nodes={n.name:n for n in ast.parse(source.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 for name in names:assert ast.dump(inline[name],include_attributes=False)==ast.dump(source_nodes[name],include_attributes=False),name
images=re.findall('data:image/png;base64,([A-Za-z0-9+/=]+)','\n'.join(c.source for c in student.cells));assert len(images)==4
for data in images:assert base64.b64decode(data).startswith(b'\x89PNG')
paths=[R/'lessons'/f'{S}.html',R/'reference/rdl-reproduction-checkpoint.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb'];before=[sha(p) for p in paths]
subprocess.run([sys.executable,str(P/'_build_l140.py')],check=True,capture_output=True);subprocess.run([sys.executable,str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True);assert before==[sha(p) for p in paths],'Builder drift'
figs=sorted((P/'figures/l140').glob('*'));before=[sha(p) for p in figs];subprocess.run([sys.executable,str(P/'_figures_l140.py')],check=True,capture_output=True);assert before==[sha(p) for p in figs]
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda msg:errors.append(msg.text) if msg.type=='error' else None)
 for width in [1200,375]:
  page.set_viewport_size(dict(width=width,height=900));page.goto((R/'lessons'/f'{S}.html').as_uri())
  box=page.locator('#l140-selection');control=box.locator('input')
  for value in [1,2,3]:
   control.fill(str(value));control.dispatch_event('input');text=box.locator('.audit-result').inner_text();assert ('Valid first maximum.' in text)==(value==2);states+=1
  box.locator('button').click();assert control.input_value()=='2';control.focus();page.keyboard.press('ArrowRight');assert control.input_value()=='3';box.locator('button').click()
  box=page.locator('#l140-gates');n=box.locator('#l140-seeds');gap=box.locator('#l140-gap');match=box.locator('#l140-match')
  for seeds in range(6):
   for delta in [0,.9,1,1.1,2]:
    for matched in [False,True]:
     n.fill(str(seeds));n.dispatch_event('input');gap.fill(str(delta));gap.dispatch_event('input');match.set_checked(matched);match.dispatch_event('input');text=box.locator('.audit-result').inner_text()
     assert ('Execution: COMPLETE.' in text)==(seeds==5)
     expected='NOT_EVALUATED' if seeds<5 else 'CLOSE' if delta<=1 else 'OUTSIDE_TOLERANCE'
     assert 'Score: '+expected+'.' in text
     assert 'Protocol: '+('ALIGNED_WITH_RELEASE' if matched else 'GAPPED')+'.' in text
     assert 'Historical identity: NOT_ESTABLISHED.' in text;states+=1
  box.locator('button').click();assert n.input_value()=='5' and gap.input_value()=='0.2' and not match.is_checked()
  gap.focus();page.keyboard.press('ArrowRight');assert gap.input_value()=='0.3';box.locator('button').click()
  assert page.locator('#warmup button').count()==0 and page.locator('#l140-teachback textarea').count()==1
  page.locator('figure img').evaluate_all('(xs)=>xs.forEach(x=>x.loading="eager")');page.wait_for_function('Array.from(document.querySelectorAll("figure img")).every(x=>x.complete && x.naturalWidth>0)');assert page.locator('figure img').evaluate_all('(xs)=>xs.length===4&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
  page.screenshot(path=f'/tmp/l140-top-{width}.png')
  for i in range(4):page.locator('figure').nth(i).screenshot(path=f'/tmp/l140-figure-{i}-{width}.png')
  page.locator('#l140-gates').screenshot(path=f'/tmp/l140-gates-{width}.png')
 page.emulate_media(media='print');page.screenshot(path='/tmp/l140-print.png',full_page=True);page.emulate_media(media='screen')
 page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('figure img[src^="data:image/png;base64,"]').count()==4
 nojs=browser.new_context(java_script_enabled=False,viewport=dict(width=375,height=900));pg=nojs.new_page();pg.goto((R/'lessons'/f'{S}.html').as_uri());assert 'epoch2' in pg.locator('noscript').first.inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');nojs.close();browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0];lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))];checked=0
with tempfile.TemporaryDirectory(prefix='l140-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/f'{S}.html',stage/'reference/rdl-reproduction-checkpoint.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme:continue
   dest=(path.parent/unquote(part.path)).resolve() if part.path else path
   assert dest.exists() and not dest.is_symlink(),(path,url);checked+=1
 for rel in ['labs/relkit/checkpoint_l140.py','labs/l140-reproduction.md','labs/solutions/'+S+'.ipynb','labs/evidence/l140/training.json','labs/_full_l140.py','modal/l140_repro.py','labs/sources/l140/examples__model.py']:
  assert (stage/rel).exists(),rel
 manifest=json.loads((stage/'lessons/manifest.json').read_text());assert next(x for x in manifest['lessons'] if x['id']==140)['labPath']=='labs/'+S+'.ipynb'
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();base=f'http://127.0.0.1:{server.server_port}'
   page.goto(base+'/index.html');link=reveal_gallery_link(page,'a[href="lessons/'+S+'.html"]');assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1;link.click();assert page.url.endswith('/lessons/'+S+'.html')
   page.goto(base+'/notebooks.html');page.wait_for_selector('a[href="labs/html/'+S+'.html"]',state='attached');browser.close()
 finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',print_nojs='PASS',portable_figures=4,live_tasks=3,executed_code_hash=digest,deterministic_rebuild='EXACT',copied_pages_links=checked,javascript_errors=errors,screenshots='/tmp/l140-*.png',full_notebook_gate='PASS: one complete trial seed100; Amazon training covered by author runs and inline AST equality, not an additional notebook fit',live_colab='NOT_CHECKED',deployment='NOT_CHECKED');(P/'_delivery_l140_results.json').write_text(json.dumps(r,indent=2));print(r)
