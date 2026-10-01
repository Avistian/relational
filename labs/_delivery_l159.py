"""Artifact, browser, copied-site and deterministic rebuild checks for L159."""
import ast,functools,hashlib,json,os,re,subprocess,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import unquote,urlsplit
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0159-foundation-model-preview'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert all(not c.outputs for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert sum(c.source.count('data:image/png;base64,') for c in solution.cells)==3
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_l159_results.json').read_text())['executed_code_sha256']
inline={n.name:n for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for n in ast.parse((P/'relkit/foundation_preview_l159.py').read_text()).body:
    if isinstance(n,(ast.FunctionDef,ast.ClassDef)):
        assert ast.dump(n,include_attributes=False)==ast.dump(inline[n.name],include_attributes=False),n.name
states=0;errors=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('console',lambda msg:errors.append(msg.text) if msg.type=='error' else None)
    page.on('requestfailed',lambda req:errors.append(req.url))
    for width in [1200,375]:
        page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/(S+'.html')).as_uri())
        host=page.locator('#mask-objective')
        for target in ['table','column','cell']:
            host.locator('[data-target]').select_option(target)
            for policy in ['global','root','cache']:
                host.locator('[data-policy]').select_option(policy)
                expected=0 if policy=='global' or (policy=='root' and target=='cell') else (1 if target=='cell' else 3 if policy=='root' else 4)
                assert host.get_attribute('data-exposed')==str(expected);states+=1
        host.locator('button').click();assert host.get_attribute('data-exposed')=='0'
        host.locator('[data-policy]').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert host.locator('[data-policy]').input_value()=='root'
        host.locator('button').click()
        gradient=page.locator('#gradient-objective')
        for value in [-1,0,.4,1]:
            gradient.locator('[data-h]').fill(str(value))
            for frozen in [False,True]:
                gradient.locator('[data-frozen]').set_checked(frozen)
                for detach in [False,True]:
                    gradient.locator('[data-detach]').set_checked(detach)
                    observed=gradient.locator('[data-grad]').inner_text()
                    if detach:assert 'Absent' in observed
                    else:
                        import math
                        assert abs(float(observed)-2*(1/(1+math.exp(-2*value))-1))<.00051
                    assert ('Absent' in gradient.locator('[data-w-grad]').inner_text())==frozen
                    states+=1
        gradient.locator('button').click();assert gradient.locator('[data-grad]').inner_text()=='-0.620'
        gradient.locator('[data-detach]').focus();page.keyboard.press('Space');assert 'Absent' in gradient.locator('[data-grad]').inner_text()
        gradient.locator('button').click()
        assert page.locator('#warmup button').count()>0
        predict=page.locator('#predict');assert predict.locator('.predict-reveal').is_disabled()
        predict.locator('[data-value="local"]').click();predict.locator('.predict-reveal').click();assert 'transfer' in predict.locator('.predict-outcome').inner_text()
        assert page.locator('#teachback textarea').count()==1
        assert page.locator('figure img').evaluate_all('(xs)=>xs.length===3&&xs.every(x=>x.complete&&x.naturalWidth>0)')
        assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
        page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l159-top-{width}.png')
        host.screenshot(path=f'/tmp/l159-mask-{width}.png');gradient.screenshot(path=f'/tmp/l159-gradient-{width}.png')
        for i in range(3):page.locator('figure').nth(i).screenshot(path=f'/tmp/l159-figure-{i}-{width}.png')
    page.emulate_media(media='print');page.pdf(path='/tmp/l159-print.pdf',format='A4');assert Path('/tmp/l159-print.pdf').stat().st_size>20000
    page.screenshot(path='/tmp/l159-print-screen.png',full_page=True)
    context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri())
    assert pg.locator('noscript').count()==2 and 'NOT_RUN' in pg.locator('body').inner_text()
    assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
    page.emulate_media(media='screen');page.set_viewport_size({'width':1000,'height':900});page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==3
    assert '100.00' in page.locator('body').inner_text()
    for i in range(3):page.locator('img[src^="data:image/png"]').nth(i).screenshot(path=f'/tmp/l159-notebook-figure-{i}.png')
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
lines=[line[10:] for line in block.splitlines() if line.startswith('          ')];lines=[line for line in lines if not line.startswith(('VER=','sed -i'))];count=0
with tempfile.TemporaryDirectory(prefix='l159-pages-') as tmp:
    stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),line) for line in lines)
    subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
    for path in [stage/'lessons'/(S+'.html'),stage/'reference/foundation-model-preview.html']:
        parser=Links();parser.feed(path.read_text())
        for url in parser.links:
            part=urlsplit(url)
            if part.scheme or not part.path:continue
            dest=(path.parent/unquote(part.path)).resolve();assert dest.exists(),str(dest);count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/foundation-model-preview.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb')]+sorted((P/'figures/l159').iterdir())
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l159.py')],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'Non-deterministic builder'
result=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',no_js='PASS',print='PASS',
    inline_source_parity='PASS',portable_figures=3,notebook_code_cells=sum(c.cell_type=='code' for c in solution.cells),
    copied_pages_links=count,deterministic_build='PASS',manifest_galleries='PASS',javascript_errors=errors,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l159_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
