"""Real browser interaction, local links, notebook source parity and deterministic builds."""
import ast,functools,hashlib,itertools,json,os,subprocess,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
from relkit.privacy_l187 import bounded_histogram,release_scale
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0187-ethics-privacy-reg'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
solcode='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');stucode='\n\n'.join(c.source for c in student.cells if c.cell_type=='code')
assert 'NotImplementedError' not in solcode and stucode.count('NotImplementedError')==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
soldefs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(solcode).body if isinstance(n,ast.FunctionDef)}
for name in ['relkit/privacy_l187.py','_check_l187.py','_run_l187.py','_verify_l187.py']:
    for n in ast.parse((P/name).read_text()).body:
        if isinstance(n,ast.FunctionDef):assert soldefs[n.name]==ast.dump(n,include_attributes=False),(name,n.name)
execution=json.loads((P/'_execution_l187_results.json').read_text())
assert execution['executed_code_sha256']==hashlib.sha256(solcode.encode()).hexdigest()
toy=pd.DataFrame({'resultId':[1,2,3,4],'driverId':[1,1,1,2],'constructorId':[10,10,20,20]})
errors=[];states=0
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
    page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('requestfailed',lambda q:errors.append(q.url))
    for width in [1200,375]:
        page.set_viewport_size({'width':width,'height':1000});page.goto((R/'lessons'/(S+'.html')).as_uri())
        host=page.locator('#privacy-removal')
        for mode,nodes,edges in [('row',6,13),('node',6,10),('entity',0,0)]:
            host.locator('select').select_option(mode);out=host.locator('output')
            assert int(out.get_attribute('data-nodes'))==nodes and int(out.get_attribute('data-edges'))==edges
            states+=1
        host.locator('[data-reset]').click();assert host.get_attribute('data-state')=='node'
        host.locator('select').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert host.get_attribute('data-state')=='entity'
        host.screenshot(path=f'/tmp/l187-removal-{width}.png');host.locator('[data-reset]').click()
        release=page.locator('#privacy-release')
        for cap,eps,repeats in itertools.product([1,2,3],[.5,1,2],[1,5,30]):
            for key,value in [('cap',cap),('epsilon',eps),('repeats',repeats)]:release.locator('[data-'+key+']').select_option(str(value))
            out=release.locator('output');actual=[int(v) for v in out.get_attribute('data-hist').split(',')]
            assert np.array_equal(actual,bounded_histogram(toy,[10,20],cap))
            expected=release_scale(cap,eps,repeats)
            assert float(out.get_attribute('data-scale'))==expected['scale'] and float(out.get_attribute('data-total'))==expected['epsilon_total']
            states+=1
        release.locator('[data-reset]').click();assert release.get_attribute('data-state')=='2/1/1'
        release.screenshot(path=f'/tmp/l187-release-{width}.png')
        pred=page.locator('#predict');assert pred.locator('.predict-reveal').is_disabled()
        assert len(set(len(s.split()) for s in pred.locator('.predict-option').all_inner_texts()))==1
        pred.locator('[data-value=node]').click();pred.locator('.predict-reveal').click();assert 'Incident edges' in pred.locator('.predict-outcome').inner_text()
        assert page.locator('#warmup button').count()==0 and page.locator('#teachback textarea').count()==1
        assert page.locator('figure img').evaluate_all('(xs)=>xs.length===4&&xs.every(x=>x.complete&&x.naturalWidth>0)')
        assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),f'Overflow {width}'
        page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l187-top-{width}.png')
        for i in range(4):page.locator('figure').nth(i).screenshot(path=f'/tmp/l187-figure-{i}-{width}.png')
    page.emulate_media(media='print');assert release.locator('.ep-controls').evaluate('(x)=>getComputedStyle(x).display')=='none'
    page.screenshot(path='/tmp/l187-print.png')
    ctx=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=ctx.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri())
    assert pg.locator('noscript').count()==2 and '1,096' in pg.locator('body').inner_text()
    assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');ctx.close()
    page.emulate_media(media='screen');page.set_viewport_size({'width':1050,'height':900});page.goto((P/'html'/(S+'.html')).as_uri())
    assert page.locator('img[src^="data:image/png"]').count()==4
    assert 'COMPLETE_COURSE_EXPERIMENT' in page.locator('body').inner_text()
    page.locator('img[src^="data:image/png"]').nth(2).screenshot(path='/tmp/l187-notebook-noise.png');browser.close()
assert not errors,errors
class Quiet(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)))
threading.Thread(target=server.serve_forever,daemon=True).start()
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
    for path,href in [('index.html','lessons/'+S+'.html'),('notebooks.html','labs/html/'+S+'.html')]:
        page.goto(f'http://127.0.0.1:{server.server_port}/'+path);reveal_gallery_link(page,'a[href="'+href+'"]')
    browser.close()
server.shutdown()
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
count=0
for path in [R/'lessons'/(S+'.html'),R/'reference/ethics-privacy-reg.html']:
    parser=Links();parser.feed(path.read_text())
    for url in parser.links:
        part=urlsplit(url)
        if part.scheme or not part.path:continue
        assert (path.parent/unquote(part.path)).resolve().is_file(),url;count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/ethics-privacy-reg.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'evidence/l187/report.md']+sorted((P/'figures/l187').iterdir())
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
for script in ['_figures_l187.py','_build_l187.py']:subprocess.run([str(R/'.venv/bin/python'),str(P/script)],check=True,capture_output=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'Nondeterministic artifacts'
result={'status':'PASS','browser_widths':[1200,375],'interactive_states':states,'python_javascript_parity':'PASS',
    'keyboard_reset':'PASS','no_js':'PASS','print':'PASS','inline_source_parity':'PASS','portable_figures':4,
    'local_links':count,'deterministic_build':'PASS','manifest_galleries':'PASS','javascript_errors':errors,
    'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/'_delivery_l187_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
