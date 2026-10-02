"""Browser, portability, source parity, deterministic build and copied-site checks."""
import ast,functools,hashlib,json,os,re,subprocess,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
from relkit.corpus_l171 import split_corpus
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0171-corpus-of-databases'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
solcode='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
stucode='\n\n'.join(c.source for c in student.cells if c.cell_type=='code')
assert solcode.count('NotImplementedError')==0 and stucode.count('NotImplementedError')==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
soldefs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(solcode).body if isinstance(n,ast.FunctionDef)}
for filename in ['relkit/corpus_l171.py','_audit_l171.py','_check_l171.py']:
    for n in ast.parse((P/filename).read_text()).body:
        if isinstance(n,ast.FunctionDef):assert soldefs[n.name]==ast.dump(n,include_attributes=False),n.name
manifest=json.loads((P/'evidence/l171/input-manifest.json').read_text())
for name,digest in manifest['files'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
report=json.loads((P/'evidence/l171/report.json').read_text());inventory=report['inventory'];errors=[];states=0
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
    page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('requestfailed',lambda req:errors.append(req.url))
    for width in [1200,375]:
        page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/(S+'.html')).as_uri());host=page.locator('#corpus-explorer')
        for row in inventory:
            host.locator('select').select_option(row['database'])
            for contaminated in [False,True]:
                host.locator('input').set_checked(contaminated)
                expected=split_corpus(inventory,row['database'])
                assert json.loads(host.get_attribute('data-train'))==expected['train']
                assert json.loads(host.get_attribute('data-quarantine'))==(['synthetic-bridge','synthetic-copy','synthetic-tail'] if contaminated else [])
                assert host.get_attribute('data-heldout')==row['database']
                assert ('3 quarantined' if contaminated else '0 quarantined') in host.locator('output').inner_text()
                assert '6 declared training candidates' in host.locator('[data-baseline]').inner_text();states+=1
        host.locator('[data-reset]').click();assert host.locator('select').input_value()=='rel-f1'
        host.locator('input').focus();page.keyboard.press('Space');assert host.locator('input').is_checked()
        host.screenshot(path=f'/tmp/l171-intervention-{width}.png')
        host.locator('[data-reset]').click();assert not host.locator('input').is_checked()
        predict=page.locator('#predict');assert predict.locator('.predict-reveal').is_disabled()
        labels=predict.locator('.predict-option').all_inner_texts();assert len(set(len(x.split()) for x in labels))==1
        predict.locator('[data-value="component"]').click();predict.locator('.predict-reveal').click();assert 'connected component' in predict.locator('.predict-outcome').inner_text()
        assert page.locator('#warmup button').count()>0 and page.locator('#teachback textarea').count()==1
        assert page.locator('figure img').evaluate_all('(xs)=>xs.length===3&&xs.every(x=>x.complete&&x.naturalWidth>0)')
        assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),'Page overflows at '+str(width)
        page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l171-top-{width}.png')
        for i in range(3):page.locator('figure').nth(i).screenshot(path=f'/tmp/l171-figure-{i}-{width}.png')
    page.emulate_media(media='print');assert host.locator('.corpus-controls').evaluate('(x)=>getComputedStyle(x).display')=='none'
    context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri())
    assert pg.locator('noscript').count()==1 and '97,606' in pg.locator('body').inner_text()
    assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
    page.emulate_media(media='screen');page.set_viewport_size({'width':1050,'height':900});page.goto((P/'html'/(S+'.html')).as_uri())
    assert page.locator('img[src^="data:image/png"]').count()==3
    assert 'COMPLETE_DECLARED_CORPUS_AUDIT' in page.locator('body').inner_text()
    page.locator('img[src^="data:image/png"]').first.screenshot(path='/tmp/l171-notebook-figure.png');browser.close()
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
with tempfile.TemporaryDirectory(prefix='l171-pages-') as tmp:
    stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),l) for l in lines)
    subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
    for path in [stage/'lessons'/(S+'.html'),stage/'reference/corpus-of-databases.html']:
        parser=Links();parser.feed(path.read_text())
        for url in parser.links:
            part=urlsplit(url)
            if part.scheme or not part.path:continue
            dest=(path.parent/unquote(part.path)).resolve();assert dest.exists(),str(dest);count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/corpus-of-databases.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'evidence/l171/report.json',P/'evidence/l171/report.md',R/'assets/l171-evidence.js']+sorted((P/'figures/l171').iterdir())
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
subprocess.run([str(R/'.venv/bin/python'),str(P/'_figures_l171.py')],check=True,capture_output=True)
subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l171.py')],check=True,capture_output=True)
after=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths];assert before==after,'Non-deterministic build'
result=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',no_js='PASS',print='PASS',inline_source_parity='PASS',portable_figures=3,notebook_code_cells=sum(c.cell_type=='code' for c in solution.cells),copied_pages_links=count,deterministic_build='PASS',manifest_galleries='PASS',javascript_errors=errors,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l171_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
