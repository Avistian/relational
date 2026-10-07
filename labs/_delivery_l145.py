"""Actual browser, portable artifact, determinism and copied Pages checks."""
import hashlib,json,os,re,subprocess,tempfile,threading,functools
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from _gallery_delivery import reveal_gallery_link
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0145-relational-graph-transformer'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);sol=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert all(not c.outputs for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in sol.cells if c.cell_type=='code')
code='\n\n'.join(c.source for c in sol.cells if c.cell_type=='code');assert 'from relkit' not in code and 'import relkit' not in code
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_l145_results.json').read_text())['executed_code_sha256']
assert sum(c.source.count('data:image/png;base64,') for c in sol.cells)==4
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('requestfailed',lambda r:errors.append(r.url))
 for width in [1200,375]:
  page.set_viewport_size(dict(width=width,height=900));page.goto((R/'lessons'/(S+'.html')).as_uri())
  host=page.locator('#relgt-token');slider=host.locator('input[type=range]');check=host.locator('input[type=checkbox]')
  for include in [True,False]:
   check.set_checked(include)
   for value in [0,3,5,10]:
    slider.fill(str(value));slider.dispatch_event('input');assert float(host.get_attribute('data-result'))==9-(value if include else 0);states+=1
  host.locator('button').click();assert slider.input_value()=='5' and check.is_checked();slider.focus();page.keyboard.press('ArrowRight');assert slider.input_value()=='6';host.locator('button').click()
  owner=page.locator('#relgt-owner');control=owner.locator('input')
  for cutoff in [5,8,9,10]:
   control.fill(str(cutoff));control.dispatch_event('input');assert owner.get_attribute('data-valid')==str(9<=cutoff).lower();states+=1
  owner.locator('button').click();assert control.input_value()=='5'
  assert page.locator('#warmup button').count()==0 and page.locator('#relgt-teachback textarea').count()==1
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===4&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
  page.screenshot(path=f'/tmp/l145-top-{width}.png');host.screenshot(path=f'/tmp/l145-token-{width}.png')
  for i in range(4):page.locator('figure').nth(i).screenshot(path=f'/tmp/l145-figure-{i}-{width}.png')
 page.emulate_media(media='print');page.pdf(path='/tmp/l145-print.pdf',format='A4');assert Path('/tmp/l145-print.pdf').stat().st_size>20000
 nojs=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=nojs.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert pg.locator('noscript').count()==4;assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');nojs.close();browser.close()
assert not errors,errors
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(SimpleHTTPRequestHandler,directory=str(R)))
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 for path,selector in [('index.html','a[href="lessons/'+S+'.html"]'),('notebooks.html','a[href="labs/html/'+S+'.html"]')]:
  page.goto('http://127.0.0.1:'+str(server.server_port)+'/'+path)
  reveal_gallery_link(page,selector)
 browser.close()
server.shutdown();server.server_close()

class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0];lines=[l[10:] for l in block.splitlines() if l.startswith('          ')];lines=[l for l in lines if not l.startswith(('VER=','sed -i'))];count=0
with tempfile.TemporaryDirectory(prefix='l145-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),l) for l in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/(S+'.html'),stage/'reference/relgt.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   dest=(path.parent/unquote(part.path)).resolve();assert dest.exists(),str(dest);count+=1
# A rebuild must preserve executed cells and all generated bytes.
paths=[R/'lessons'/(S+'.html'),R/'reference/relgt.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb')]
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths];subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l145.py')],check=True,capture_output=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
report=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',no_js='PASS',print='PASS',portable_figures=4,notebook_code_cells=sum(c.cell_type=='code' for c in sol.cells),copied_pages_links=count,deterministic_build='PASS',manifest_galleries='PASS',live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l145_results.json').write_text(json.dumps(report,indent=2));print(report)
