"""Verify live learner work, source refusal, measured browser controls and links."""
import base64,copy,functools,http.server,json,os,re,shutil,subprocess,sys,tempfile,threading,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
P=Path(__file__).resolve().parent;R=P.parent;V=R/'reviews/lesson-b18';S='b18-context-sufficiency';start=time.monotonic();V.mkdir(parents=True,exist_ok=True)
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
r=json.loads((P/'evidence/b18/summary.json').read_text())
student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
indices=[i for i,c in enumerate(student.cells) if c.cell_type=='code' and 'raise NotImplementedError("Implement' in c.source]
assert len(indices)==3 and all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
images=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)','\n'.join(c.source for c in student.cells));assert len(images)==3
assert all(base64.b64decode(x).startswith(b'\x89PNG\r\n\x1a\n') for x in images)
for index in indices:
    n=copy.deepcopy(student)
    for j in indices:
        if j!=index:n.cells[j].source=solution.cells[j].source
    with tempfile.TemporaryDirectory(prefix='b18-blank-') as td:
        try:NotebookClient(n,timeout=90,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
        except CellExecutionError as e:assert 'NotImplementedError' in str(e)
        else:raise AssertionError('Blank task passed')
# Deliberately wrong learner operations must fail their behavioral checks.
import _test_b18 as tests
mutants={'eligible_history':lambda rows,cutoff:rows,'corrected_total':lambda sample,n:float(sum(sample)),'stratified_metrics':lambda d,p,y:{}}
for name,bad in mutants.items():
    old=getattr(tests,name);setattr(tests,name,bad)
    try:
        tests.Boundaries({'eligible_history':'test_history','corrected_total':'test_correction','stratified_metrics':'test_metrics'}[name]).debug()
    except (AssertionError,KeyError):pass
    else:raise AssertionError('Incorrect learner operation passed: '+name)
    finally:setattr(tests,name,old)
with tempfile.TemporaryDirectory(prefix='b18-gate-') as td:
    t=Path(td);shutil.copytree(P/'sources/b18',t/'sources/b18');shutil.copy(P/'_reproduce_b18.py',t/'_reproduce_b18.py')
    good=subprocess.run([sys.executable,str(t/'_reproduce_b18.py'),'--phase','audit'],capture_output=True,text=True);assert good.returncode==0
    blocked=subprocess.run([sys.executable,str(t/'_reproduce_b18.py'),'--phase','paper'],capture_output=True,text=True);assert blocked.returncode!=0 and 'NOT_RUN:' in blocked.stderr
    (P/'evidence/b18/paper-refusal.txt').write_text(blocked.stdout+blocked.stderr)
    f=t/'sources/b18/paper.html';f.write_bytes(f.read_bytes()+b'corrupted')
    bad=subprocess.run([sys.executable,str(t/'_reproduce_b18.py'),'--phase','audit'],capture_output=True,text=True);assert bad.returncode!=0 and 'SOURCE_HASH_MISMATCH' in bad.stderr
class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}/';errors=[];states=0
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('requestfailed',lambda req:errors.append(req.url));page.on('response',lambda response:errors.append(response.url) if response.url.startswith(base) and response.status>=400 else None)
    for width in [1200,375]:
        page.set_viewport_size({'width':width,'height':950});page.goto(base+f'lessons/{S}.html')
        board=page.locator('[data-context-budget]')
        for c in r['conditions']:
            for name in ['seed','degree','shape','budget']:board.locator(f'[name={name}]').select_option(str(c[name]))
            assert abs(float(board.locator('output').get_attribute('data-corrected-rmse'))-c['metrics']['corrected']['rmse'])<1e-9
            assert board.locator('output').get_attribute('data-key')==f"{c['seed']}/{c['degree']}/{c['shape']}/{c['budget']}";states+=1
        board.locator('button').click();assert board.locator('[name=budget]').input_value()=='32'
        board.locator('[name=budget]').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert board.locator('[name=budget]').input_value()=='128';board.locator('button').click()
        temporal=page.locator('#b18-time');slider=temporal.locator('input')
        for rule in ['event','static']:
            temporal.locator('select').select_option(rule)
            for cutoff in range(11):
                slider.fill(str(cutoff));slider.dispatch_event('input');assert f'Day {cutoff}' in temporal.locator('.temporal-readout').inner_text();states+=1
        temporal.locator('button').click();assert 'Legal mean: 3.000' in temporal.locator('.temporal-readout').inner_text()
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2'),'Horizontal page overflow'
        assert page.locator('#b18-warmup').inner_text().strip();assert page.locator('#b18-predict').inner_text().strip();assert page.locator('#b18-teachback textarea').count()==1
        page.screenshot(path=str(V/f'lesson-{width}.png'),full_page=True)
        board.screenshot(path=str(V/f'budget-{width}.png'));temporal.screenshot(path=str(V/f'time-{width}.png'));page.locator('[aria-label="Information pipeline"]').screenshot(path=str(V/f'pipeline-{width}.png'))
    page.emulate_media(media='print');assert not board.locator('button').is_visible();assert board.locator('output').is_visible();page.screenshot(path=str(V/'print.png'),full_page=True);page.emulate_media(media='screen')
    nojs=browser.new_page(java_script_enabled=False);nojs.goto(base+f'lessons/{S}.html');assert nojs.locator('noscript').first.is_visible();assert nojs.locator('table').count()>0;nojs.close()
    page.goto(base+'index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]')
    page.goto(base+'notebooks.html');page.wait_for_selector('a[href="labs/'+S+'.ipynb"]')
    page.set_viewport_size({'width':1000,'height':900});page.goto(base+f'labs/html/{S}.html');assert page.locator('img').count()==3
    for i in range(3):page.locator('img').nth(i).screenshot(path=str(V/f'notebook-{i}.png'))
    browser.close()
server.shutdown();assert not errors,errors
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=set()
    def handle_starttag(self,t,a):
        self.links.extend(v for k,v in a if k in ('href','src'));self.ids.update(v for k,v in a if k in ('id','name'))
count=0
for name in ['lessons/'+S+'.html','reference/'+S+'.html']:
    path=R/name;p=Links();p.feed(path.read_text())
    for url in p.links:
        u=urlsplit(url)
        if u.scheme:continue
        target=(path.parent/unquote(u.path)).resolve() if u.path else path
        assert target.is_file(),url;count+=1
        if u.fragment and target.suffix=='.html':
            other=Links();other.feed(target.read_text());assert u.fragment in other.ids,url
out=dict(status='PASS',live_blank_tasks_rejected=3,wrong_learner_functions_rejected=3,embedded_pngs=3,portable_source_gate='PASS',corrupt_source_rejected=True,paper_attempt='REFUSED_AS_REQUIRED',browser_states=states,keyboard_reset='PASS',desktop_mobile='PASS',print_nojs='PASS',local_links=count,browser_errors=errors,seconds=time.monotonic()-start,live_colab='NOT_CHECKED',deployment='NOT_REQUESTED')
(P/'_delivery_b18_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
