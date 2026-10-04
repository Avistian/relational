"""Check real learner work, source gate and desktop/mobile browser delivery."""
import copy,functools,http.server,json,os,re,shutil,subprocess,sys,tempfile,threading,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
P=Path(__file__).resolve().parent;R=P.parent;V=R/'reviews/lesson-b18a';S='b18a-context-state';started=time.monotonic()
student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
indices=[i for i,c in enumerate(student.cells) if c.cell_type=='code' and 'raise NotImplementedError("Implement' in c.source]
assert len(indices)==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
for index in indices:
    n=copy.deepcopy(student)
    for j in indices:
        if j!=index:n.cells[j].source=solution.cells[j].source
    with tempfile.TemporaryDirectory(prefix='b18a-blank-') as td:
        try:NotebookClient(n,timeout=90,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
        except CellExecutionError as e:assert 'NotImplementedError' in str(e)
        else:raise AssertionError('Blank task passed')
import _test_b18a as tests
mutants={'eligible_support':lambda rows,cutoff:rows,'state_key':lambda **kwargs:'0'*64,'replacement_contrast':lambda predict,x,background,feature:predict(x)}
for name,bad in mutants.items():
    old=getattr(tests,name);setattr(tests,name,bad)
    try:tests.StateContract({'eligible_support':'test_temporal_labels','state_key':'test_identity','replacement_contrast':'test_contrast'}[name]).debug()
    except AssertionError:pass
    else:raise AssertionError('Wrong operation accepted: '+name)
    finally:setattr(tests,name,old)
with tempfile.TemporaryDirectory(prefix='b18a-source-') as td:
    t=Path(td);shutil.copytree(P/'sources/b18a',t/'sources/b18a');shutil.copy(P/'_reproduce_b18a.py',t/'_reproduce_b18a.py')
    good=subprocess.run([sys.executable,str(t/'_reproduce_b18a.py'),'--phase','audit'],capture_output=True,text=True);assert good.returncode==0
    blocked=subprocess.run([sys.executable,str(t/'_reproduce_b18a.py'),'--phase','paper'],capture_output=True,text=True);assert blocked.returncode and 'INCOMPLETE_SOURCE_PROTOCOL_GATE' in blocked.stderr
    (V/'paper-refusal.txt').write_text(blocked.stdout+blocked.stderr)
    f=t/'sources/b18a/paper.html';f.write_bytes(f.read_bytes()+b'changed')
    bad=subprocess.run([sys.executable,str(t/'_reproduce_b18a.py')],capture_output=True,text=True);assert bad.returncode and 'SOURCE_HASH_MISMATCH' in bad.stderr
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}/';errors=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('response',lambda r:errors.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
    for width in [1200,375]:
        page.set_viewport_size(dict(width=width,height=950));page.goto(base+f'lessons/{S}.html');panel=page.locator('[data-context-state]')
        for value in ['none','support','preprocess','query','background']:
            panel.locator('select').select_option(value);assert panel.locator('output').get_attribute('data-state')==value
        assert 'Support cache: Reuse' in panel.locator('output').inner_text() and 'Explanation result: Recompute' in panel.locator('output').inner_text()
        panel.locator('button').click();panel.locator('select').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert panel.locator('select').input_value()=='support'
        panel.locator('button').click();assert panel.locator('select').input_value()=='none'
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2'),'Page overflows'
        if width==375:
            diagram=page.locator('.state-figure').first;diagram.focus();page.keyboard.press('ArrowRight');page.wait_for_timeout(200)
            assert diagram.evaluate('e=>e.scrollLeft>0'),'Diagram keyboard scroll failed'
            diagram.evaluate('e=>e.scrollLeft=0')
        page.screenshot(path=str(V/f'lesson-{width}.png'),full_page=True);panel.screenshot(path=str(V/f'widget-{width}.png'))
        page.locator('.state-figure').first.screenshot(path=str(V/f'architecture-{width}.png'))
    page.emulate_media(media='print');assert not panel.locator('select').is_visible();assert panel.locator('output').is_visible();page.screenshot(path=str(V/'print.png'),full_page=True);page.emulate_media(media='screen')
    nojs=browser.new_page(java_script_enabled=False);nojs.goto(base+f'lessons/{S}.html');assert nojs.locator('noscript').is_visible();assert nojs.locator('table').count()>=3;nojs.close()
    page.goto(base+'index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]')
    page.goto(base+'notebooks.html');page.wait_for_selector('a[href="labs/'+S+'.ipynb"]')
    page.goto(base+f'labs/html/{S}.html');assert page.locator('img').count()==3
    browser.close()
server.shutdown();assert not errors,errors
class Links(HTMLParser):
    def __init__(self):super().__init__();self.urls=[];self.ids=set()
    def handle_starttag(self,t,a):
        self.urls.extend(v for k,v in a if k in ['href','src']);self.ids.update(v for k,v in a if k in ['id','name'])
count=0
for file in [R/'lessons'/f'{S}.html',R/'reference'/f'{S}.html']:
    links=Links();links.feed(file.read_text())
    for url in links.urls:
        u=urlsplit(url)
        if u.scheme:continue
        target=(file.parent/unquote(u.path)).resolve() if u.path else file
        assert target.is_file(),url;count+=1
        if u.fragment and target.suffix=='.html':
            parser=Links();parser.feed(target.read_text());assert u.fragment in parser.ids,url
r=dict(status='PASS',blank_tasks_rejected=3,wrong_functions_rejected=3,source_corruption_rejected=True,paper_gate_refused=True,browser_states=10,mobile_desktop_keyboard_print_nojs='PASS',local_links=count,browser_errors=errors,seconds=time.monotonic()-started)
(P/'_delivery_b18a_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
