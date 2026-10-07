"""Browser, standalone notebook, navigation and copied-Pages integrity checks."""
import hashlib,json,os,re,subprocess,tempfile,threading,functools
from pathlib import Path
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0148-ablation-discipline'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);sol=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert all(not c.outputs for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in sol.cells if c.cell_type=='code')
assert sum(c.source.count('data:image/png;base64,') for c in sol.cells)==4
code='\n\n'.join(c.source for c in sol.cells if c.cell_type=='code')
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_l148_results.json').read_text())['executed_code_sha256']
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/(S+'.html')).as_uri())
  host=page.locator('#history_widget');window=host.locator('select')
  for days in ['30','365','730']:
   window.select_option(days);assert host.get_attribute('data-window')==days;assert 'Future' in host.locator('tbody').inner_text();states+=1
  host.locator('button').click();assert window.input_value()=='365'
  window.focus();page.keyboard.press('ArrowUp');assert window.input_value()=='30';host.locator('button').click()
  inter=page.locator('#interaction_widget');slider=inter.locator('input')
  for value,expected in [('5','-2'),('6.5','-0.5'),('7','0'),('8','1'),('9','2')]:
   slider.fill(value);slider.dispatch_event('input');assert inter.get_attribute('data-interaction')==expected;states+=1
  inter.locator('button').click();assert slider.input_value()=='6.5'
  slider.focus();page.keyboard.press('ArrowRight');assert inter.get_attribute('data-interaction')=='0';inter.locator('button').click()
  predict=page.locator('#predict');assert predict.locator('.predict-reveal').is_disabled()
  predict.locator('[data-value="interact"]').click();predict.locator('.predict-reveal').click();assert 'interaction' in predict.locator('.predict-outcome').inner_text()
  assert page.locator('#warmup button').count()==0 and page.locator('#teachback textarea').count()==1
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===4&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
  if width==375:
   fig=page.locator('figure').first;fig.focus();page.keyboard.press('ArrowRight');page.wait_for_timeout(150);assert fig.evaluate('(e)=>e.scrollLeft>0');fig.evaluate('(e)=>e.scrollLeft=0')
  page.screenshot(path=f'/tmp/l148-top-{width}.png');host.screenshot(path=f'/tmp/l148-map-{width}.png')
  for i in range(4):page.locator('figure').nth(i).screenshot(path=f'/tmp/l148-figure-{i}-{width}.png')
 page.emulate_media(media='print');page.pdf(path='/tmp/l148-print.pdf',format='A4');assert Path('/tmp/l148-print.pdf').stat().st_size>20000
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert pg.locator('noscript').count()==4;assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 # Inspect rendered notebook figure payloads and its generated result table.
 page.emulate_media(media='screen');page.goto((P/'html'/(S+'.html')).as_uri())
 assert page.locator('img[src^="data:image/png"]').count()==4
 assert 'Verified author predictions: 31475' in page.locator('body').inner_text()
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
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0];lines=[l[10:] for l in block.splitlines() if l.startswith('          ')];lines=[l for l in lines if not l.startswith(('VER=','sed -i'))];count=0
with tempfile.TemporaryDirectory(prefix='l148-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),l) for l in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/(S+'.html'),stage/'reference/ablation-discipline.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   dest=(path.parent/unquote(part.path)).resolve();assert dest.exists(),str(dest);count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/ablation-discipline.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb')]
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths];subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l148.py')],check=True,capture_output=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',no_js='PASS',print='PASS',portable_figures=4,notebook_code_cells=sum(c.cell_type=='code' for c in sol.cells),copied_pages_links=count,deterministic_build='PASS',manifest_galleries='PASS',javascript_errors=errors,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l148_results.json').write_text(json.dumps(r,indent=2));print(r)
