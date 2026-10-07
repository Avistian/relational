"""Actual browser interventions, artifact integrity and copied Pages delivery."""
import ast,hashlib,json,os,re,subprocess,tempfile,threading,functools
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from _gallery_delivery import reveal_gallery_link
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0146-gnn-vs-graph-transformer'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);sol=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert all(not c.outputs for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in sol.cells if c.cell_type=='code')
for cell in sol.cells:
 if cell.cell_type=='code':
  for node in ast.walk(ast.parse(cell.source)):
   if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith('relkit')
code='\n\n'.join(c.source for c in sol.cells if c.cell_type=='code');assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_l146_results.json').read_text())['executed_code_sha256']
assert sum(c.source.count('data:image/png;base64,') for c in sol.cells)==4
errors=[];states=0;geometry=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('requestfailed',lambda r:errors.append(r.url))
 for width in [1200,375]:
  page.set_viewport_size(dict(width=width,height=900));page.goto((R/'lessons'/(S+'.html')).as_uri())
  host=page.locator('#cmp-path');slider=host.locator('input[type=range]');include=host.locator('.included');future=host.locator('.future')
  for inc in [True,False]:
   include.set_checked(inc)
   for fut in [True,False]:
    future.set_checked(fut)
    for depth in [1,2,3]:
     slider.fill(str(depth));slider.dispatch_event('input')
     assert host.get_attribute('data-gnn')==str(inc and not fut and depth>=3).lower()
     assert host.get_attribute('data-transformer')==str(inc and not fut).lower();states+=1
  host.locator('button').click();assert slider.input_value()=='2' and include.is_checked() and not future.is_checked()
  slider.focus();page.keyboard.press('ArrowRight');assert slider.input_value()=='3';host.locator('button').click()
  selection=page.locator('#cmp-selection');control=selection.locator('select')
  for rule,expected in [('validation','L4/.3'),('test','L1/.5')]:
   control.select_option(rule);assert selection.get_attribute('data-selected')==expected;assert selection.locator('tr.selected').count()==1;states+=1
  selection.locator('button').click();assert control.input_value()=='validation'
  control.focus();page.keyboard.press('ArrowDown');assert control.input_value()=='test';selection.locator('button').click()
  predict=page.locator('#cmp-predict');assert predict.locator('.predict-reveal').is_disabled()
  predict.locator('[data-value="reversal"]').click();predict.locator('.predict-reveal').click()
  assert 'GNN wins all three test pairs' in predict.locator('.predict-outcome').inner_text();states+=1
  assert page.locator('#warmup button').count()==0 and page.locator('#cmp-teachback textarea').count()==1
  page.locator('figure img').evaluate_all('(xs)=>xs.forEach(x=>x.loading="eager")')
  page.wait_for_function('Array.from(document.querySelectorAll("figure img")).every(x=>x.complete&&x.naturalWidth>0)')
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===5&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
  page.screenshot(path=f'/tmp/l146-top-{width}.png');host.screenshot(path=f'/tmp/l146-path-{width}.png');selection.screenshot(path=f'/tmp/l146-select-{width}.png')
  for i in range(5):page.locator('figure').nth(i).screenshot(path=f'/tmp/l146-figure-{i}-{width}.png')
 page.emulate_media(media='print');page.pdf(path='/tmp/l146-print.pdf',format='A4');assert Path('/tmp/l146-print.pdf').stat().st_size>20000
 nojs=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=nojs.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert pg.locator('noscript').count()==4;assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');nojs.close()
 # Geometry checks use SVG native coordinates; background excluded from box overlaps.
 for name in ['paths','selection','architecture','results']:
  page.goto((P/f'figures/l146/{name}.svg').as_uri())
  result=page.evaluate('''()=>{let svg=document.querySelector('svg'),v=svg.viewBox.baseVal,bad=[];for(let n of svg.querySelectorAll('text,rect,circle')){let b=n.getBBox();if(b.x<-.1||b.y<-.1||b.x+b.width>v.width+.1||b.y+b.height>v.height+.1)bad.push(n.textContent||n.tagName);}let boxes=[...svg.querySelectorAll('rect')].slice(1).map(x=>x.getBBox());for(let i=0;i<boxes.length;i++)for(let j=i+1;j<boxes.length;j++){let a=boxes[i],b=boxes[j];if(a.x<b.x+b.width&&b.x<a.x+a.width&&a.y<b.y+b.height&&b.y<a.y+a.height)bad.push('box overlap');}return bad;}''')
  assert not result,(name,result);geometry+=1
 browser.close()
assert not errors,errors
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(SimpleHTTPRequestHandler,directory=str(R)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
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
with tempfile.TemporaryDirectory(prefix='l146-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),l) for l in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 for path in [stage/'lessons'/(S+'.html'),stage/'reference/gnn-vs-graph-transformer.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   dest=(path.parent/unquote(part.path)).resolve();assert dest.exists(),str(dest);count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/gnn-vs-graph-transformer.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb')]
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths];subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l146.py')],check=True,capture_output=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',no_js='PASS',print='PASS',page_figures=5,portable_figures=4,svg_geometry_figures=geometry,notebook_code_cells=sum(c.cell_type=='code' for c in sol.cells),copied_pages_links=count,deterministic_build='PASS',manifest_galleries='PASS',live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l146_results.json').write_text(json.dumps(r,indent=2));print(r)
