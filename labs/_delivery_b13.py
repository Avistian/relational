"""Exercise blank learner gates, immutable replay, browser states and local links."""
import functools,http.server,json,os,re,subprocess,tempfile,threading,time,zipfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
from _audit_b13 import audit_l200
P=Path(__file__).resolve().parent;R=P.parent;V=R/'reviews/lesson-b13';S='b13-synthetic-relational-data';start=time.monotonic();V.mkdir(parents=True,exist_ok=True)
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/f'{S}.ipynb',4)
assert sum('raise NotImplementedError("Complete' in c.source for c in student.cells if c.cell_type=='code')==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert all(not re.search(r'\{\{[A-Z_]+\}\}',c.source) for c in student.cells if c.cell_type=='markdown')
with tempfile.TemporaryDirectory(prefix='b13-blank-') as td:
    try:NotebookClient(student,timeout=90,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
    except CellExecutionError as e:assert 'Complete canonical_schema' in str(e)
    else:raise AssertionError('Blank student passed')
gate=subprocess.run([str(R/'.venv/bin/python'),str(P/'_reproduce_b13.py'),'--run-full'],capture_output=True,text=True)
assert gate.returncode!=0 and 'BLOCKED: USD0' in gate.stderr
mutations=0
with tempfile.TemporaryDirectory(prefix='b13-corrupt-') as td:
    with zipfile.ZipFile(P/'evidence/b13/l200-reproducer.zip') as z:z.extractall(td)
    root=Path(td)/'labs/evidence/l200'
    for name in ['pilot-1/RDBPFN-0.npz','packet/prepared.npz','full-1/receipt.json']:
        file=root/name;original=file.read_bytes();file.write_bytes(original+b'corruption')
        try:audit_l200(root)
        except (ValueError,AssertionError):mutations+=1
        else:raise AssertionError('Changed inherited evidence admitted')
        finally:file.write_bytes(original)
class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*a):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}/';errors=[];states=0
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('response',lambda r:errors.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
    for width in [1200,375]:
        page.set_viewport_size({'width':width,'height':950});page.goto(base+f'lessons/{S}.html')
        board=page.locator('[data-fk-trace]')
        for a in range(2):
            for b in range(2):
                board.locator('[name=a]').select_option(str(a));board.locator('[name=b]').select_option(str(b));mean=([2,6][a]+[10,14][b])/2
                assert float(board.locator('output').get_attribute('data-mean'))==mean
                assert float(board.locator('output').get_attribute('data-prediction'))==2+mean;states+=1
        board.locator('button').click();assert board.locator('[name=a]').input_value()=='0' and board.locator('[name=b]').input_value()=='1'
        board.locator('[name=a]').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert board.locator('[name=a]').input_value()=='1';board.locator('button').click()
        schema=page.locator('[data-schema-identity]')
        for family in ['chain','out-star','in-star','diamond']:
            for rename in ['no','yes']:
                schema.locator('[name=family]').select_option(family);schema.locator('[name=rename]').select_option(rename)
                assert schema.locator('output').get_attribute('data-heldout')==str(family=='diamond').lower()
                assert schema.locator('svg line').count()==(4 if family=='diamond' else 3)
                assert schema.locator('svg circle').count()==4
                labels=schema.locator('svg text').all_text_contents();assert labels==(['A','B','C','D'] if rename=='no' else ['Z','X','W','Y']);states+=1
        schema.locator('button').click();assert schema.locator('[name=family]').input_value()=='chain'
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2'),'Horizontal overflow'
        assert page.locator('#b13-warmup').inner_text().strip()
        assert page.locator('#b13-predict').inner_text().strip()
        assert page.locator('#b13-teachback textarea').count()==1
        assert page.locator('img').evaluate_all('(images)=>images.every(x=>x.complete&&x.naturalWidth>0)')
        if width==375:
            assert page.locator('[data-mobile-results]').is_visible()
            assert page.locator('[data-mobile-results] li').count()==12
            page.locator('[data-mobile-results]').screenshot(path=str(V/'results-375.png'))
        page.screenshot(path=str(V/f'lesson-{width}.png'),full_page=True)
        page.locator('.synthetic-mobile' if width==375 else '.synthetic-wide').first.screenshot(path=str(V/f'architecture-{width}.png'))
        board.screenshot(path=str(V/f'trace-{width}.png'));schema.screenshot(path=str(V/f'schema-{width}.png'))
    page.emulate_media(media='print');assert page.locator('.synthetic-wide').first.is_visible();assert not page.locator('[data-fk-trace] button').is_visible()
    page.emulate_media(media='screen')
    nojs=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});q=nojs.new_page();q.goto(base+f'lessons/{S}.html');assert 'isomorphic' in q.locator('body').inner_text();assert 'Controls need JavaScript' in q.locator('body').inner_text();nojs.close()
    page.goto(base+'index.html');page.wait_for_timeout(400);assert page.locator('a[href="lessons/'+S+'.html"]').count()>=1
    page.goto(base+'notebooks.html');page.wait_for_timeout(400);assert S in page.content()
    page.goto(base+f'labs/html/{S}.html');assert page.locator('img').count()>=4
    browser.close()
server.shutdown();assert not errors,errors
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,t,a):self.links.extend(v for k,v in a if k in ('href','src'))
links=0
for name in ['lessons/'+S+'.html','reference/'+S+'.html']:
    path=R/name;parser=Links();parser.feed(path.read_text())
    for url in parser.links:
        u=urlsplit(url)
        if u.scheme or not u.path:continue
        assert (path.parent/unquote(u.path)).resolve().is_file(),url;links+=1
result=dict(status='PASS',blank_student='REJECTED',historical_evidence_mutations_rejected=mutations,interactive_states=states,desktop_mobile_keyboard_reset='PASS',print_nojs='PASS',local_links=links,manifest_galleries='PASS',browser_errors=errors,seconds=time.monotonic()-start,live_colab='NOT_CHECKED')
(P/'_delivery_b13_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
