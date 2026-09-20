"""Real browser interactions, deterministic rebuild and copied Pages delivery."""
import hashlib,json,os,re,subprocess,tempfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0095-bipartite-graphs'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
    page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
    for width in [1200,375]:
        page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/f'{S}.html').as_uri())
        assert page.locator('#warmup').inner_text().strip()
        walk=page.locator('#walk-widget');box=walk.locator('input');out=walk.locator('output')
        assert '0.125' in out.inner_text();box.check();assert '= 0.000' in out.inner_text()
        walk.locator('button').click();assert not box.is_checked() and '= 0.125' in out.inner_text()
        box.focus();page.keyboard.press('Space');assert '= 0.000' in out.inner_text();walk.locator('button').click()
        boundary=page.locator('#boundary-widget');assert boundary.locator('output').inner_text().startswith('PASS')
        boundary.locator('input').check();assert boundary.locator('output').inner_text().startswith('FAIL')
        boundary.locator('button').click();assert boundary.locator('output').inner_text().startswith('PASS')
        assert page.locator('#prediction button').count()>=3 and page.locator('#teachback textarea').count()==1
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Page overflow'
        for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
        page.screenshot(path=f'/tmp/l095-top-{width}.png');walk.screenshot(path=f'/tmp/l095-widget-{width}.png')
        page.locator('figure').nth(1).screenshot(path=f'/tmp/l095-pipeline-{width}.png')
        if width==375:
            sc=walk.locator('.figure-scroll');sc.focus();page.keyboard.press('ArrowRight');page.wait_for_timeout(180);assert sc.evaluate('(e)=>e.scrollLeft>0')
    page.set_viewport_size({'width':1200,'height':900});page.emulate_media(media='print');assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1');page.emulate_media(media='screen')
    page.goto((P/'html'/f'{S}.html').as_uri());assert page.locator('img[src^="data:image/png;base64,"]').count()==4
    for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
    page.locator('img').nth(2).screenshot(path='/tmp/l095-notebook-walk.png')
    nojs=browser.new_page(java_script_enabled=False);nojs.goto((R/'notebooks.html').as_uri());assert nojs.locator('#lab-95 a').count()==4
    browser.close()
assert not errors,errors
student=nbformat.read(P/f'{S}.ipynb',as_version=4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',as_version=4)
assert sum('raise NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==3
assert not any(c.get('outputs') for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None for c in solution.cells if c.cell_type=='code')
assert 'attachment:' not in json.dumps(student)
assert [c.source for c in student.cells if c.cell_type=='code' and not c.metadata.get('task')]==[c.source for c in solution.cells if c.cell_type=='code' and not c.metadata.get('task')]
paths=[R/'lessons'/f'{S}.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb',P/'html'/f'{S}.html']
def hashes():return [hashlib.sha256(x.read_bytes()).hexdigest() for x in paths]
before=hashes();subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l095.py')],check=True,capture_output=True);assert hashes()==before,'Rebuild changed artifacts'
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
(P/'_delivery_l095_results.json').write_text('{"status":"RUNNING"}\n')
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0];lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))]
with tempfile.TemporaryDirectory(prefix='l095-pages-') as tmp:
    stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
    subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
    checked=0
    for path in [stage/'lessons'/f'{S}.html',stage/'reference/bipartite-contract.html']:
        parser=Links();parser.feed(path.read_text())
        for url in parser.links:
            part=urlsplit(url)
            if part.scheme or not part.path:continue
            dest=(path.parent/unquote(part.path)).resolve();assert dest.exists(),(path,url);assert not dest.is_symlink();checked+=1
    for relative in ['labs/relkit/bipartite_l095.py','labs/_run_l095.py','labs/l095-reproduction.md','labs/results/l095/fold-5.json','labs/solutions/'+S+'.ipynb']:
        assert (stage/relative).exists(),relative
    import functools,threading
    from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(QuietHandler,directory=str(stage)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
            page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}/index.html')
            page.wait_for_selector('a[href="lessons/'+S+'.html"]')
            assert page.locator('a[href="labs/html/'+S+'.html"]').count()>=1
            page.goto(f'http://127.0.0.1:{server.server_port}/lessons/{S}.html')
            assert '0.125' in page.locator('#walk-widget output').inner_text()
            browser.close()
    finally:server.shutdown();server.server_close();thread.join()
assert not errors,errors
result={'status':'PASS','browser_widths':[1200,375],'widget_states_checked':4,'keyboard_reset_print':'PASS','portable_figures':4,'solution_code_cells':18,'copied_pages_local_links':checked,'deterministic_rebuild':'EXACT','manifest_navigation_over_http':'PASS','javascript_errors':errors,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/'_delivery_l095_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
