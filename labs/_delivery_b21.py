"""Exercise real learner tasks, source gates and desktop/mobile browser states."""
import copy,functools,http.server,json,os,re,shutil,subprocess,sys,tempfile,threading,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
P=Path(__file__).resolve().parent;R=P.parent;V=R/'reviews/lesson-b21';S='b21-structural-robustness';started=time.monotonic()
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
indices=[i for i,c in enumerate(student.cells) if c.cell_type=='code' and c.metadata.get('learner_function')];assert len(indices)==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert len(re.findall('data:image/png;base64,','\n'.join(c.source for c in student.cells)))==4
for index in indices:
    n=copy.deepcopy(student)
    for j in indices:
        if j!=index:n.cells[j].source=solution.cells[j].source
    with tempfile.TemporaryDirectory(prefix='b21-blank-') as td:
        try:NotebookClient(n,timeout=90,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
        except CellExecutionError as e:assert 'NotImplementedError' in str(e)
        else:raise AssertionError('Blank task passed')
# Execute actual notebook CHECK cells with a plausible wrong implementation, not separate proxy tests.
wrong={'changed_cells':"def changed_cells(original, assignment):\n    return int(np.any(np.asarray(original)!=np.asarray(assignment)))",
'validate_assignment':"def validate_assignment(data, assignment, budget):\n    return True",
'direction_score':"def direction_score(gf, gr, before, after):\n    return float(np.sum(np.asarray(gf)*np.asarray(after)))"}
for index in indices:
    n=copy.deepcopy(solution)
    name=n.cells[index].metadata['learner_function'];n.cells[index].source=wrong[name]
    with tempfile.TemporaryDirectory(prefix='b21-wrong-') as td:
        try:NotebookClient(n,timeout=90,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
        except CellExecutionError as e:assert 'AssertionError' in str(e)
        else:raise AssertionError('Wrong task passed '+name)
with tempfile.TemporaryDirectory(prefix='b21-source-gate-') as td:
    t=Path(td);shutil.copytree(P/'sources/b21',t/'sources/b21');shutil.copy(P/'_reproduce_b21.py',t/'_reproduce_b21.py')
    proc=subprocess.run([sys.executable,str(t/'_reproduce_b21.py'),'--fresh'],capture_output=True,text=True);assert proc.returncode!=0 and 'NOT_RUN:' in proc.stderr
    f=t/'sources/b21/paper.html';f.write_bytes(f.read_bytes()+b'corrupt')
    proc=subprocess.run([sys.executable,str(t/'_reproduce_b21.py')],capture_output=True,text=True);assert proc.returncode!=0 and 'SOURCE_HASH_MISMATCH' in proc.stderr
class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}/';errors=[];states=0
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
    page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('requestfailed',lambda req:errors.append(req.url));page.on('response',lambda r:errors.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
    for width in [1200,375]:
        page.set_viewport_size({'width':width,'height':950});page.goto(base+f'lessons/{S}.html');board=page.locator('[data-fk-board]')
        for unit in ['session','single','other']:
            for dest in range(5):
                for budget in range(3):
                    board.locator('[name=unit]').select_option(unit);board.locator('[name=dest]').select_option(str(dest));board.locator('[name=budget]').select_option(str(budget))
                    a=[0,0,1,2,1,2];original=a.copy()
                    for i in {'session':[0,1],'single':[0],'other':[2]}[unit]:a[i]=dest
                    cost=sum(x!=y for x,y in zip(a,original))
                    valid=dest<3 and a[0]==a[1] and cost<=budget
                    out=board.locator('output');assert out.get_attribute('data-valid')==str(valid).lower()
                    assert f'{cost} FK cells changed; allowance {budget}' in out.inner_text()
                    assert board.locator('.changed').count()==cost
                    values=[1,3,2,5,4,6]
                    means=[]
                    for p in [0,1,2]:
                        v=[values[i] for i,x in enumerate(a) if x==p];means.append(sum(v)/len(v) if v else 0)
                    assert ', '.join(f'{x:.2f}' for x in means) in board.locator('[data-preview]').inner_text();states+=1
        board.locator('button').click();assert board.locator('[name=unit]').input_value()=='session';assert board.locator('[name=dest]').input_value()=='1';assert board.locator('[name=budget]').input_value()=='2'
        board.locator('[name=budget]').focus();page.keyboard.press('ArrowUp');page.keyboard.press('Enter');assert board.locator('[name=budget]').input_value()=='1';assert board.locator('output').get_attribute('data-valid')=='false';board.locator('button').click()
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2'),'Page horizontal overflow'
        assert page.locator('#b21-warmup').inner_text().strip()
        page.screenshot(path=str(V/f'lesson-{width}.png'),full_page=True);board.screenshot(path=str(V/f'fk-{width}.png'))
        for i,fig in enumerate(page.locator('.fk-figure').all()):
            if width==375:
                assert fig.evaluate('(el)=>el.scrollWidth>el.clientWidth');fig.focus();page.keyboard.press('ArrowRight');fig.evaluate('(el)=>el.scrollLeft=0')
            fig.screenshot(path=str(V/f'figure-{i}-{width}.png'))
        for choice in ['no','yes']:
            page.goto(base+f'lessons/{S}.html');pred=page.locator('#b21-predict');assert pred.locator('.predict-reveal').is_disabled();pred.locator('[data-value='+choice+']').click();pred.locator('.predict-reveal').click();assert pred.locator('.predict-outcome').inner_text().strip();states+=1
        teach=page.locator('#b21-teachback');teach.locator('textarea').fill('Existing IDs, shared session owners and event-time eligibility require separate checks. Count two changed cells for a coupled move, regenerate both graph directions and verify the same clean predictor before attacks. Exhaustive bounds only apply within the finite threat set.');teach.locator('button').first.click();assert 'An existing ID' in teach.inner_text()
        page.goto(base+f'labs/html/{S}.html');assert page.locator('img[src^="data:image/png"]').count()>=4;assert page.evaluate('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)')
        for i,fig in enumerate(page.locator('.fk-figure').all()):fig.screenshot(path=str(V/f'notebook-figure-{i}-{width}.png'))
    page.goto(base+'index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]');page.goto(base+'notebooks.html');page.wait_for_selector('a[href="labs/'+S+'.ipynb"]')
    page.goto(base+f'lessons/{S}.html');page.emulate_media(media='print');assert page.locator('.fk-controls').is_hidden();assert page.locator('.fk-figure img').first.evaluate('(el)=>getComputedStyle(el).minWidth')=='0px';page.screenshot(path=str(V/'print.png'),full_page=True)
    context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});p=context.new_page();p.goto(base+f'lessons/{S}.html');assert 'ADMISSIBLE' in p.locator('output').inner_text();assert p.locator('noscript').inner_text();p.screenshot(path=str(V/'nojs.png'),full_page=True);browser.close()
server.shutdown();assert not errors,errors
class Links(HTMLParser):
    def __init__(self):super().__init__();self.urls=[]
    def handle_starttag(self,tag,attrs):self.urls.extend(v for k,v in attrs if k in ('href','src'))
links=0
for file in [R/'lessons'/f'{S}.html',R/'reference'/f'{S}.html']:
    parser=Links();parser.feed(file.read_text())
    for url in parser.urls:
        u=urlsplit(url)
        if u.scheme:continue
        target=(file.parent/unquote(u.path)).resolve() if u.path else file
        assert target.is_file(),url;links+=1
out=dict(status='PASS',blank_tasks_rejected=3,wrong_notebook_tasks_rejected=3,source_corruption_rejected=True,historical_gate=True,browser_states=states,desktop_and_375px=True,keyboard_reset=True,no_js=True,print=True,portable_figures=4,local_links=links,browser_errors=errors,seconds=time.monotonic()-started)
(P/'_delivery_b21_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
