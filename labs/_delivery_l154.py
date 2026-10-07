"""Browser, notebook parity, deterministic generation and copied-site link checks."""
import ast,functools,hashlib,json,os,re,subprocess,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0154-portfolio-synthesis'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert all(not c.outputs for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert sum(c.source.count('data:image/png;base64,') for c in solution.cells)==2
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_l154_results.json').read_text())['executed_code_sha256']
inline={n.name:n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
for file,names in [('relkit/portfolio_l154.py',['summarize_runs','compare_entries','portfolio_verdict']),('_replay_l154.py',['aligned_score','verify_inputs','replay','render_report'])]:
 nodes={n.name:n for n in ast.parse((P/file).read_text()).body if isinstance(n,ast.FunctionDef)}
 for name in names:assert ast.dump(nodes[name],include_attributes=False)==ast.dump(inline[name],include_attributes=False),name
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/(S+'.html')).as_uri())
  host=page.locator('#l154-evidence');metric=host.locator('[data-metric]');origin=host.locator('[data-origin]');units=host.locator('[data-units]');complete=host.locator('[data-complete]')
  for m in ['MAE','AUROC']:
   metric.select_option(m)
   for o in ['published','local']:
    origin.select_option(o)
    for u in ['display','raw']:
     units.select_option(u)
     for c in [True,False]:
      complete.set_checked(c)
      expected=(1 if m=='MAE' else (10 if u=='display' else .1))
      assert abs(float(host.get_attribute('data-gap'))-expected)<1e-10
      actual=host.locator('[data-claim]').inner_text()
      assert ('INCOMPLETE' if not c else 'PUBLISHED_CONTEXT_ONLY' if o=='published' else 'LOCAL_MATCHED_DESCRIPTIVE') in actual
      assert host.locator('[data-coverage]').inner_text().startswith('1/1' if c else '0/1');states+=1
  host.locator('button').click();assert metric.input_value()=='MAE' and origin.input_value()=='published' and complete.is_checked()
  complete.focus();page.keyboard.press('Space');assert not complete.is_checked();host.locator('button').click()
  assert page.locator('#warmup button').count()==0
  predict=page.locator('#predict');assert predict.locator('.predict-reveal').is_disabled()
  predict.locator('[data-value="context"]').click();predict.locator('.predict-reveal').click();assert 'fresh matched baseline' in predict.locator('.predict-outcome').inner_text()
  assert page.locator('#teachback textarea').count()==1
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===2&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
  page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l154-top-{width}.png');host.screenshot(path=f'/tmp/l154-widget-{width}.png')
  for i in range(2):page.locator('figure').nth(i).screenshot(path=f'/tmp/l154-figure-{i}-{width}.png')
 page.emulate_media(media='print');page.pdf(path='/tmp/l154-print.pdf',format='A4');assert Path('/tmp/l154-print.pdf').stat().st_size>20000
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert pg.locator('noscript').count()==1;assert '2/3' in pg.locator('body').inner_text() or 'two' in pg.locator('body').inner_text().lower();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.goto((P/'html'/(S+'.html')).as_uri())
 assert page.locator('img[src^="data:image/png"]').count()==2
 assert 'Coverage: 2/3 tasks' in page.locator('body').inner_text()
 browser.close()
assert not errors,errors
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start()
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 for path,selector in [('index.html','a[href="lessons/'+S+'.html"]'),('notebooks.html','a[href="labs/html/'+S+'.html"]')]:
  page.goto('http://127.0.0.1:'+str(server.server_port)+'/'+path);reveal_gallery_link(page,selector)
 browser.close()
server.shutdown();server.server_close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[l[10:] for l in block.splitlines() if l.startswith('          ')];lines=[l for l in lines if not l.startswith(('VER=','sed -i'))];count=0
with tempfile.TemporaryDirectory(prefix='l154-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),l) for l in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/(S+'.html'),stage/'reference/portfolio-synthesis.html',stage/'labs/html'/(S+'.html')]:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   dest=(path.parent/unquote(part.path)).resolve();assert dest.exists(),str(dest);count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/portfolio-synthesis.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'evidence/l154/report.json',P/'evidence/l154/report.md']+sorted((P/'figures/l154').iterdir())
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l154.py')],check=True,capture_output=True)
subprocess.run(['python3',str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'Builder is not deterministic'
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',no_js='PASS',print='PASS',
       inline_source_parity='PASS',portable_figures=2,notebook_code_cells=sum(c.cell_type=='code' for c in solution.cells),
       copied_pages_links=count,deterministic_build='PASS',manifest_galleries='PASS',javascript_errors=errors,
       live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l154_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
